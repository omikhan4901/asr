"""
Whisper model loader and configuration manager with robust tokenization & prompt IDs.
"""

import re
from typing import Dict, List
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

# Per-batch generate() warnings (max_length vs max_new_tokens, duplicate logits
# processors, tokenizer cleanup) are harmless and flood the batch logs.
hf_logging.set_verbosity_error()

from bangla_asr.config import WhisperConfig


_MARKERS = re.compile(r"<\|[^|>]*\|>")


def strip_markers(text: str) -> str:
    """Remove Whisper control tokens (<|bn|>, <|notimestamps|>, ...) left in decoded text.

    Some fine-tuned checkpoints (e.g. BanglaASR) do not register them as special
    tokens, so skip_special_tokens=True does not always remove them.
    """
    return _MARKERS.sub("", text).strip()


def bangla_generation_config(cfg: WhisperConfig, processor, model):
    """Generation config with the Bangla transcribe prompt, plus the kwargs to pass to generate().

    Shared by inference (WhisperASRModel) and fine-tuning (pipeline/11_finetune.py)
    so both decode with the same prompt.
    """
    # Prefer the fine-tuned checkpoint's settings. Some regional
    # checkpoints ship an older Whisper generation_config.json, however,
    # which lacks lang_to_id/task_to_id and cannot accept language/task
    # under modern Transformers. Keep a modern base config available to
    # fill only those compatibility fields.
    try:
        gen_config = GenerationConfig.from_pretrained(cfg.model_id)
    except Exception:
        gen_config = model.generation_config

    try:
        modern_base_config = GenerationConfig.from_pretrained(cfg.fallback_base_config)
    except Exception:
        modern_base_config = None

    if modern_base_config is not None:
        for attribute in ("lang_to_id", "task_to_id", "is_multilingual", "no_timestamps_token_id"):
            if not hasattr(gen_config, attribute) and hasattr(modern_base_config, attribute):
                setattr(gen_config, attribute, getattr(modern_base_config, attribute))

    # Configure the Whisper prompt directly on a generation config which
    # has the modern language/task mappings. Do not pass language/task as
    # generate() kwargs: older fine-tuned configs reject those kwargs
    # before generation begins.
    prompt_kwargs = {}
    if hasattr(gen_config, "lang_to_id") and hasattr(gen_config, "task_to_id"):
        gen_config.language = cfg.language
        gen_config.task = cfg.task
        # Also pass them to generate(): current transformers ignores
        # generation_config.language for multilingual checkpoints such as
        # openai/whisper-large-v3-turbo (it auto-detects and wrote Bangla
        # speech in Devanagari). lang_to_id is guaranteed present here.
        prompt_kwargs = {"language": cfg.language, "task": cfg.task}
        if hasattr(gen_config, "forced_decoder_ids"):
            gen_config.forced_decoder_ids = None
    else:
        # Compatibility fallback for an offline setup where the base
        # generation config could not be downloaded.
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
        except (AttributeError, ValueError):
            forced_decoder_ids = None
        if hasattr(gen_config, "forced_decoder_ids"):
            gen_config.forced_decoder_ids = forced_decoder_ids
    return gen_config, prompt_kwargs


class WhisperASRModel:
    """Encapsulates WhisperProcessor and WhisperForConditionalGeneration with verified Bangla prompt IDs."""

    def __init__(self, config: WhisperConfig = WhisperConfig(), device: str = None):
        self.cfg = config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Loading Whisper model: {self.cfg.model_id} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(self.cfg.model_id)

        # Load FP16 on GPU, float32 on CPU
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = {"torch_dtype": torch_dtype}
        if self.device.type == "cuda" and self.cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.attn_implementation

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            **model_kwargs,
        )
        if self.cfg.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading a LoRA adapter requires peft. Install with: pip install peft"
                ) from exc
            print(f"Loading LoRA adapter from {self.cfg.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path)
        self.model = self.model.to(self.device)

        gen_config, self.prompt_kwargs = bangla_generation_config(self.cfg, self.processor, self.model)
        gen_config.num_beams = self.cfg.num_beams
        gen_config.num_return_sequences = self.cfg.num_beams
        gen_config.early_stopping = True
        gen_config.length_penalty = self.cfg.length_penalty
        # Do NOT default output_scores/return_dict_in_generate to True here.
        # BatchedNBestGenerator needs sequences_scores and sets both flags
        # itself in its own gen_kwargs (self-contained). transcribe_batch()
        # below calls generate() and hands the result straight to
        # batch_decode() expecting a plain token-id tensor -- if this shared,
        # mutable generation_config defaulted return_dict_in_generate to True,
        # it would silently get
        # a GenerateOutput object instead and batch_decode() would iterate
        # its fields (sequences, then scores, ...) rather than decode tokens,
        # surfacing as "Boolean value of Tensor with more than one value is
        # ambiguous" deep inside the tokenizer.

        if self.cfg.diverse_beam_search:
            gen_config.do_sample = True
            gen_config.temperature = self.cfg.sampling_temperature
        else:
            gen_config.do_sample = False

        # With a LoRA adapter, generate() runs on the wrapped base model and reads its config.
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        base.generation_config = gen_config
        self.model.eval()
        print("Whisper model successfully loaded and configured.")

    def _text_ids(self, texts: List[str]):
        """Token ids of each text after the Bangla transcribe prompt, and the prompt length."""
        tok = self.processor.tokenizer
        tok.set_prefix_tokens(language=self.cfg.language, task=self.cfg.task, predict_timestamps=False)
        return [tok(t).input_ids for t in texts], len(tok.prefix_tokens)

    def count_text_tokens(self, texts: List[str]) -> List[int]:
        """n_tokens that score_texts() reports for each text (text tokens + end-of-text), without the GPU."""
        ids, prefix = self._text_ids(texts)
        return [len(x) - prefix for x in ids]

    @torch.no_grad()
    def score_texts(self, audio, texts: List[str], sampling_rate: int = 16000) -> List[Dict]:
        """Exact log-probability of each text given one audio (teacher forcing).

        audio: 1-D array of one utterance (standard 30 s Whisper input). Each text is
        tokenized after the Bangla transcribe prompt and scored including the
        end-of-text token. Returns mean_logprob (per token), sum_logprob and
        n_tokens per text. Used for N-best candidate scores, because current
        transformers no longer returns sequences_scores / beam_indices for
        Whisper beam search. A text longer than the decoder's position limit
        (448 tokens incl. the prompt; a decoded text can re-tokenize longer than
        it was generated) is scored on the longest prefix that fits and flagged
        "truncated" (its end-of-text token is not scored).
        """
        tok = self.processor.tokenizer
        ids, prefix = self._text_ids(texts)
        limit = self.model.config.max_target_positions + 1  # decoder input = every token but the last
        cut = [len(x) > limit for x in ids]
        ids = [x[:limit] for x in ids]
        L = max(len(x) for x in ids)
        pad = tok.eos_token_id
        seq = torch.full((len(ids), L), pad, dtype=torch.long)
        for k, x in enumerate(ids):
            seq[k, : len(x)] = torch.tensor(x)
        seq = seq.to(self.device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        feats = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        enc = self.model.get_encoder()(feats.to(self.device, dtype=dtype)).last_hidden_state
        enc = enc.expand(len(ids), -1, -1)
        logits = self.model(encoder_outputs=(enc,), decoder_input_ids=seq[:, :-1]).logits.float()
        lp = torch.log_softmax(logits, -1).gather(-1, seq[:, 1:, None])[..., 0]
        out = []
        for k, x in enumerate(ids):
            v = lp[k, prefix - 1 : len(x) - 1]  # text tokens + end-of-text
            out.append({"mean_logprob": float(v.mean()), "sum_logprob": float(v.sum()), "n_tokens": int(v.numel()),
                        **({"truncated": True} if cut[k] else {})})
        return out

    def transcribe_batch(
        self,
        audio_arrays,
        sampling_rate: int = 16000,
        max_new_tokens: int = None,
        pad_seconds: float = 0.2,
        num_beams: int = 1,
    ):
        """1-best transcription: greedy (num_beams=1, the default everywhere) or beam search.
        Default cap 440 new tokens (Whisper allows 448 including prompt tokens); dense 25 s chunks
        reach ~250 tokens."""
        from bangla_asr.common.audio import pad_silence

        if not audio_arrays:
            return []

        padded = [
            pad_silence(arr, pad_seconds, sampling_rate) if pad_seconds > 0 else arr
            for arr in audio_arrays
        ]
        inputs = self.processor(
            padded,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        input_features = inputs.input_features.to(self.device, dtype=torch_dtype)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        gen_kwargs = {
            "input_features": input_features,
            "max_new_tokens": max_new_tokens or 440,
            "num_beams": num_beams,
            "num_return_sequences": 1,
            "do_sample": False,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(self.prompt_kwargs)

        with torch.no_grad():
            gen_ids = self.model.generate(**gen_kwargs)
        return [strip_markers(t) for t in self.processor.batch_decode(gen_ids, skip_special_tokens=True)]
