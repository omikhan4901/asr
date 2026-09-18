"""
Thin generation wrapper for the correction policy (default: VibeThinker-3B).

This is a scaffold: it loads a causal LM and returns text. It does not
fine-tune, does not browse a clinical KB, and does not claim VibeThinker
is already a good Bangla medical editor.
"""

from __future__ import annotations

from typing import Optional

import torch

from bangla_asr.config import GRPOConfig


class ReasoningModel:
    """Config-swappable chat/completion model used by the ReAct scaffold and GRPO."""

    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        device: Optional[str] = None,
        *,
        model=None,
        tokenizer=None,
    ) -> None:
        self.cfg = config or GRPOConfig()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            self._load()

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading reasoning model {self.cfg.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Greedy completion used by the ReAct stub and held-out eval."""
        assert self.model is not None and self.tokenizer is not None
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
