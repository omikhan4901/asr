"""
``ASRModel`` adapter over the existing Whisper loader and batched N-best generator.

This file does not change generate kwargs or MBR-facing candidate dicts. It only
converts the historical ``{"text", "logprob", "rank"}`` records into ``Hypothesis``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from bangla_asr.config import AudioConfig, WhisperConfig
from bangla_asr.models.autoregressive.generator import BatchedNBestGenerator
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
from bangla_asr.models.base import ASRModel, Hypothesis


def hypotheses_from_candidate_dicts(candidates: List[Dict]) -> List[Hypothesis]:
    """Convert a generator/JSONL candidate list into interface hypotheses."""
    hyps: List[Hypothesis] = []
    for cand in candidates:
        hyps.append(
            Hypothesis(
                text=str(cand.get("text", "")),
                logprob=float(cand.get("logprob", 0.0)),
                rank=int(cand.get("rank", len(hyps))),
            )
        )
    return hyps


class WhisperModel(ASRModel):
    """Whisper N-best model that preserves the published beam-search path.

    When ``n`` equals ``WhisperConfig.num_beams`` (12 in the baseline), decoding
    is delegated to ``BatchedNBestGenerator.generate_batch`` with no overrides.
    A different ``n`` requests that many beams for this call only.
    """

    model_family = "whisper"

    def __init__(
        self,
        config: Optional[WhisperConfig] = None,
        device: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
        *,
        wrapper: Optional[WhisperASRModel] = None,
        generator: Optional[BatchedNBestGenerator] = None,
    ) -> None:
        self.cfg = config or WhisperConfig()
        self.audio_cfg = audio_config or AudioConfig()
        if wrapper is not None:
            self.wrapper = wrapper
        else:
            self.wrapper = WhisperASRModel(self.cfg, device=device)
        self.generator = generator or BatchedNBestGenerator(
            self.wrapper,
            audio_config=self.audio_cfg,
            whisper_config=self.cfg,
        )

    def transcribe_nbest(
        self,
        audio: np.ndarray,
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[Hypothesis]:
        """Decode one utterance to ``n`` scored Whisper hypotheses."""
        if n < 1:
            raise ValueError("n must be >= 1")
        batch = self.transcribe_nbest_batch([audio], n, sampling_rate=sampling_rate)
        return batch[0] if batch else []

    def transcribe_nbest_batch(
        self,
        audios: Sequence[np.ndarray],
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[List[Hypothesis]]:
        """Batched N-best that reuses the existing generator implementation."""
        if n < 1:
            raise ValueError("n must be >= 1")
        if sampling_rate != self.audio_cfg.sample_rate:
            # The published generator always extracts features at AudioConfig.sample_rate.
            # Resampling is the caller's responsibility (pipeline load_audio already does this).
            raise ValueError(
                f"WhisperModel expected sampling_rate={self.audio_cfg.sample_rate}, "
                f"got {sampling_rate}. Resample before calling transcribe_nbest."
            )
        num_candidates = None if n == self.cfg.num_beams else n
        raw_batches = self.generator.generate_batch(list(audios), num_candidates=num_candidates)
        return [hypotheses_from_candidate_dicts(cands) for cands in raw_batches]
