"""
GRPO training loop for a small reasoning policy that corrects ASR output.

Why GRPO (not MGPO): one group of G on-policy rollouts per prompt, advantages
are relative inside that group, and there is no critic. That matches a 3B
policy on a single Modal GPU. Multi-objective / multi-group (MGPO) is a later
axis once SWER itself is frozen.

The reward is ``swer_improvement_reward`` — never inlined — so it can be
validated on hand-labeled pairs before a GPU job is launched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bangla_asr.common.metrics import compute_wer
from bangla_asr.common.seed import set_seed
from bangla_asr.config import GRPOConfig, SWERConfig
from bangla_asr.metrics.swer import corpus_swer, is_empty_reference
from bangla_asr.training.callbacks import EarlyStopping, JsonCsvLogger, save_checkpoint
from bangla_asr.training.reward import swer_improvement_reward


def compute_group_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize rewards within each prompt group. Shape: (batch, group)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim == 1:
        rewards = rewards[None, :]
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    advantages = (rewards - mean) / std
    # Flat groups (all equal rewards) must not invent a ranking.
    flat = rewards.std(axis=1, keepdims=True) < eps
    advantages = np.where(flat, 0.0, advantages)
    return advantages


def build_simple_prompt(asr_hypothesis: str) -> str:
    """Instruction-only correction prompt (default GRPO style)."""
    return (
        "নিচের স্বয়ংক্রিয় বাংলা প্রতিলিপিটি সংশোধন করুন। "
        "শুধু সংশোধিত বাক্যটি লিখুন, অন্য কিছু নয়।\n\n"
        f"ASR: {asr_hypothesis}\n\n"
        "সংশোধন:"
    )


def build_correction_prompt(asr_hypothesis: str, prompt_style: str = "simple") -> str:
    """Dispatch to the simple prompt or the ReAct scaffold when requested."""
    if prompt_style == "react":
        from bangla_asr.correction.react_scaffold import build_react_prompt

        return build_react_prompt(asr_hypothesis)
    return build_simple_prompt(asr_hypothesis)


def load_correction_records(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load ``(asr_hypothesis, ground_truth)`` pairs from JSONL.

    Accepts either dedicated correction rows or published N-best records
    (rank-0 / greedy_text becomes the ASR hypothesis).
    """
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            hyp = raw.get("asr_hypothesis")
            if hyp is None:
                hyp = raw.get("greedy_text")
            if hyp is None:
                cands = raw.get("candidates") or []
                if cands:
                    hyp = min(cands, key=lambda c: c.get("rank", 0)).get("text", "")
            gt = raw.get("ground_truth", raw.get("gt", ""))
            if hyp is None or not str(gt).strip():
                continue
            records.append(
                {
                    "utt_id": raw.get("utt_id", ""),
                    "asr_hypothesis": str(hyp),
                    "ground_truth": str(gt),
                }
            )
    return records


def evaluate_corrections(
    asr_hypotheses: Sequence[str],
    corrections: Sequence[str],
    references: Sequence[str],
    *,
    reward_scale: float,
    reward_clip: float,
    swer_config: Optional[SWERConfig] = None,
    exclude_empty_reference: bool = True,
) -> Dict[str, float]:
    """Held-out SWER/WER/reward snapshot used by the eval callback."""
    rewards: List[float] = []
    excluded = 0
    kept_asr: List[str] = []
    kept_corr: List[str] = []
    kept_ref: List[str] = []
    for asr, corr, ref in zip(asr_hypotheses, corrections, references):
        value = swer_improvement_reward(
            asr,
            corr,
            ref,
            scale=reward_scale,
            clip=reward_clip,
            swer_config=swer_config,
            exclude_empty_reference=exclude_empty_reference,
        )
        if value is None:
            excluded += 1
            continue
        rewards.append(value)
        kept_asr.append(asr)
        kept_corr.append(corr)
        kept_ref.append(ref)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "excluded_empty_reference": float(excluded),
        "n_rewarded": float(len(rewards)),
        "asr_wer": float(compute_wer(kept_asr, kept_ref)) if kept_ref else float("nan"),
        "corr_wer": float(compute_wer(kept_corr, kept_ref)) if kept_ref else float("nan"),
        "asr_swer": float(corpus_swer(kept_asr, kept_ref, config=swer_config)) if kept_ref else float("nan"),
        "corr_swer": float(corpus_swer(kept_corr, kept_ref, config=swer_config)) if kept_ref else float("nan"),
    }


class GRPOTrainer:
    """On-policy GRPO trainer around a HuggingFace causal LM.

    ``model`` / ``ref_model`` / ``tokenizer`` may be injected in tests.
    The default constructor loads ``GRPOConfig.model_id``.
    """

    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        swer_config: Optional[SWERConfig] = None,
        *,
        model=None,
        ref_model=None,
        tokenizer=None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = config or GRPOConfig()
        self.swer_cfg = swer_config or SWERConfig()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        if self.model is None:
            self._load_policy()
        self.logger = JsonCsvLogger(self.cfg.log_dir)
        self.early_stopping = EarlyStopping(patience=self.cfg.early_stopping_patience, mode="max")

    def _load_policy(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading GRPO policy {self.cfg.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id)
        self.model.to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def generate_group(self, prompt: str) -> List[str]:
        """Sample ``group_size`` on-policy completions for one prompt."""
        assert self.model is not None and self.tokenizer is not None
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **encoded,
            do_sample=True,
            temperature=self.cfg.sampling_temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            num_return_sequences=self.cfg.group_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        texts = []
        for seq in outputs:
            texts.append(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip())
        return texts

    def completion_logprobs(self, prompt: str, completion: str, model) -> torch.Tensor:
        """Sum token log-probs of ``completion`` under ``model`` (teacher-forced)."""
        assert self.tokenizer is not None
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        full_ids = self.tokenizer(prompt + completion, add_special_tokens=True, return_tensors="pt")
        input_ids = full_ids["input_ids"].to(self.device)
        prompt_len = prompt_ids["input_ids"].shape[1]
        outputs = model(input_ids=input_ids)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        completion_logp = token_logp[:, prompt_len - 1 :]
        return completion_logp.sum(dim=-1)

    def train(
        self,
        train_records: Sequence[Dict[str, Any]],
        eval_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run GRPO for ``max_steps``. Returns the last eval snapshot."""
        if not train_records:
            raise ValueError("train_records is empty")
        set_seed(self.cfg.seed)
        assert self.model is not None
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        last_eval: Dict[str, Any] = {}

        for step in range(1, self.cfg.max_steps + 1):
            batch = [train_records[(step - 1 + i) % len(train_records)] for i in range(self.cfg.batch_size)]
            loss, stats = self._grpo_step(batch)
            if loss is None:
                row = {"step": step, "loss": float("nan"), **stats}
                self.logger.log(row)
                print(
                    f"[GRPO] step={step} skipped "
                    f"(excluded_empty_reference={stats.get('excluded_empty_reference', 0)})"
                )
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()

            row = {"step": step, "loss": float(loss.detach().cpu()), **stats}
            self.logger.log(row)
            print(
                f"[GRPO] step={step} loss={row['loss']:.4f} reward={stats['mean_reward']:.4f} "
                f"excluded_empty_reference={int(stats.get('excluded_empty_reference', 0))}"
            )

            if eval_records and step % self.cfg.eval_every == 0:
                last_eval = self._run_eval(eval_records, step)
                if self.early_stopping.update(last_eval["mean_reward"]):
                    print(f"[GRPO] early stopping at step {step}")
                    break

            if step % self.cfg.checkpoint_every == 0:
                self._save_model(step, last_eval)

        self._save_model("final", last_eval)
        return last_eval

    def _grpo_step(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        exclude = self.cfg.exclude_empty_reference_from_batch
        usable = []
        excluded = 0
        for rec in batch:
            if exclude and is_empty_reference(rec["ground_truth"]):
                excluded += 1
                continue
            usable.append(rec)
        if not usable:
            return None, {
                "mean_reward": float("nan"),
                "reward_std": float("nan"),
                "kl": 0.0,
                "excluded_empty_reference": float(excluded),
                "n_rewarded": 0.0,
            }

        advantages_all: List[np.ndarray] = []
        logprob_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        rewards_flat: List[float] = []
        rollout_excluded = 0

        for rec in usable:
            prompt = build_correction_prompt(rec["asr_hypothesis"], self.cfg.prompt_style)
            completions = self.generate_group(prompt)
            scored: List[float] = []
            kept_completions: List[str] = []
            for text in completions:
                value = swer_improvement_reward(
                    rec["asr_hypothesis"],
                    text,
                    rec["ground_truth"],
                    scale=self.cfg.reward_scale,
                    clip=self.cfg.reward_clip,
                    swer_config=self.swer_cfg,
                    exclude_empty_reference=exclude,
                )
                if value is None:
                    rollout_excluded += 1
                    continue
                scored.append(float(value))
                kept_completions.append(text)
            if not scored:
                excluded += 1
                continue
            rewards = np.asarray(scored, dtype=np.float64)
            rewards_flat.extend(float(r) for r in rewards)
            advantages = compute_group_advantages(rewards[None, :], eps=self.cfg.advantage_eps)[0]
            advantages_all.append(advantages)
            for text, adv in zip(kept_completions, advantages):
                logp = self.completion_logprobs(prompt, text, self.model)
                logprob_terms.append(logp * float(adv))
                if self.ref_model is not None and self.cfg.kl_coef > 0:
                    with torch.no_grad():
                        ref_logp = self.completion_logprobs(prompt, text, self.ref_model)
                    kl_terms.append(logp.detach() * 0.0 + (logp - ref_logp))

        if not logprob_terms:
            return None, {
                "mean_reward": float("nan"),
                "reward_std": float("nan"),
                "kl": 0.0,
                "excluded_empty_reference": float(excluded + rollout_excluded),
                "n_rewarded": 0.0,
            }

        policy_loss = -torch.stack(logprob_terms).mean()
        if kl_terms:
            kl = torch.stack(kl_terms).mean()
            loss = policy_loss + self.cfg.kl_coef * kl
            kl_value = float(kl.detach().cpu())
        else:
            loss = policy_loss
            kl_value = 0.0
        stats = {
            "mean_reward": float(np.mean(rewards_flat)),
            "reward_std": float(np.std(rewards_flat)),
            "kl": kl_value,
            "excluded_empty_reference": float(excluded + rollout_excluded),
            "n_rewarded": float(len(rewards_flat)),
        }
        return loss, stats

    def _run_eval(self, eval_records: Sequence[Dict[str, Any]], step: int) -> Dict[str, Any]:
        assert self.model is not None
        self.model.eval()
        asr, corr, refs = [], [], []
        with torch.no_grad():
            for rec in eval_records:
                prompt = build_correction_prompt(rec["asr_hypothesis"], self.cfg.prompt_style)
                encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                out = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.cfg.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                prompt_len = encoded["input_ids"].shape[1]
                text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
                asr.append(rec["asr_hypothesis"])
                corr.append(text)
                refs.append(rec["ground_truth"])
        self.model.train()
        metrics = evaluate_corrections(
            asr,
            corr,
            refs,
            reward_scale=self.cfg.reward_scale,
            reward_clip=self.cfg.reward_clip,
            swer_config=self.swer_cfg,
            exclude_empty_reference=self.cfg.exclude_empty_reference_from_batch,
        )
        row = {"step": step, "split": "eval", **metrics}
        self.logger.log(row)
        print(
            f"[GRPO eval] step={step} reward={metrics['mean_reward']:.4f} "
            f"SWER {metrics['asr_swer']:.4f}->{metrics['corr_swer']:.4f}"
        )
        return metrics

    def _save_model(self, tag: Any, eval_metrics: Dict[str, Any]) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir) / f"step_{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(ckpt_dir)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(ckpt_dir)
        save_checkpoint(
            {"step": tag, "model_id": self.cfg.model_id, "eval": eval_metrics},
            str(ckpt_dir),
            "meta",
        )


def run_training_from_jsonl(
    train_jsonl: str,
    eval_jsonl: Optional[str] = None,
    config: Optional[GRPOConfig] = None,
) -> Dict[str, Any]:
    """CLI-facing helper used by the pipeline runner and the Modal job."""
    trainer = GRPOTrainer(config=config)
    train_records = load_correction_records(train_jsonl)
    eval_records = load_correction_records(eval_jsonl) if eval_jsonl else None
    return trainer.train(train_records, eval_records)
