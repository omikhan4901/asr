"""
Shared acoustic-model interface so rescoring, SWER, and RL stay model-agnostic.

Both Whisper (autoregressive beam search) and Wav2Vec2 (CTC beam search) must
emit the same hypothesis type. Downstream MBR still consumes the historical
JSONL wire format ``{"text", "logprob", "rank"}`` via ``Hypothesis.to_candidate_dict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class Hypothesis:
    """One N-best item in the format ``MBRSelector.select`` already consumes.

    ``logprob`` is a higher-is-better sequence score. Whisper uses the
    length-normalized sequence log-probability from ``generate``. CTC beam
    search uses the ``pyctcdecode`` beam score. These are **not** on the same
    numeric scale; MBR temperature must be calibrated per model family.
    """

    text: str
    logprob: float
    rank: int
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_candidate_dict(self) -> Dict[str, Any]:
        """Exact wire format expected by ``MBRSelector.select`` and JSONL dumps."""
        return {"text": self.text, "logprob": self.logprob, "rank": self.rank}


class CTCBeamSearchUnavailableError(RuntimeError):
    """Raised when CTC N-best is requested but ``pyctcdecode`` is not installed.

    Greedy CTC decoding is only valid for ``n == 1``. Silently repeating the
    1-best transcript ``n`` times would invalidate MBR and oracle headroom.
    """


class ASRModel(ABC):
    """Acoustic model that can emit a scored N-best list."""

    model_family: str

    @abstractmethod
    def transcribe_nbest(
        self,
        audio: np.ndarray,
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[Hypothesis]:
        """Return up to ``n`` scored hypotheses, best-first (rank 0 = 1-best).

        Implementations must not pad a 1-best result to length ``n``. If
        N-best decoding is unavailable, raise rather than repeating greedy.
        """

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int = 16000,
    ) -> Hypothesis:
        """Convenience 1-best wrapper around ``transcribe_nbest(..., n=1)``."""
        hyps = self.transcribe_nbest(audio, n=1, sampling_rate=sampling_rate)
        if not hyps:
            return Hypothesis(text="", logprob=float("-inf"), rank=0)
        return hyps[0]

    def transcribe_nbest_batch(
        self,
        audios: Sequence[np.ndarray],
        n: int,
        *,
        sampling_rate: int = 16000,
    ) -> List[List[Hypothesis]]:
        """Default sequential batching. Whisper overrides this with true batching."""
        return [
            self.transcribe_nbest(audio, n, sampling_rate=sampling_rate)
            for audio in audios
        ]
