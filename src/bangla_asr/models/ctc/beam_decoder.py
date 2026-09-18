"""
CTC beam-search N-best via ``pyctcdecode``, with optional KenLM.

This module never falls back to greedy decoding when N-best is requested.
Missing ``pyctcdecode`` (or a missing KenLM file that was explicitly configured)
raises ``CTCBeamSearchUnavailableError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from bangla_asr.models.base import CTCBeamSearchUnavailableError


class _DecodeBeamsFn(Protocol):
    def decode_beams(self, logits: np.ndarray, beam_width: int) -> Sequence: ...


def require_pyctcdecode() -> None:
    """Import-check used before any N-best CTC decode (n > 1)."""
    try:
        import pyctcdecode  # noqa: F401
    except ImportError as exc:
        raise CTCBeamSearchUnavailableError(
            "CTC N-best requires pyctcdecode. Install with: pip install 'bangla_asr[ctc]' "
            "or: pip install pyctcdecode. Greedy 1-best remains available without it."
        ) from exc


class CTCBeamDecoder:
    """Thin wrapper around ``pyctcdecode.BeamSearchDecoderCTC``.

    ``decoder`` may be injected in tests so N-best behavior can be checked
    without downloading weights or a KenLM binary.
    """

    def __init__(
        self,
        labels: Sequence[str],
        kenlm_model_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        *,
        decoder: Optional[_DecodeBeamsFn] = None,
    ) -> None:
        self.labels = list(labels)
        self.kenlm_model_path = kenlm_model_path
        self.alpha = alpha
        self.beta = beta
        if decoder is not None:
            self._decoder = decoder
            return

        require_pyctcdecode()
        from pyctcdecode import build_ctcdecoder

        lm_path = kenlm_model_path
        if lm_path:
            if not Path(lm_path).exists():
                raise FileNotFoundError(
                    f"KenLM model path was configured but does not exist: {lm_path}"
                )
        self._decoder = build_ctcdecoder(
            self.labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta,
        )

    def decode_nbest(self, logits: np.ndarray, n: int) -> List[Tuple[str, float]]:
        """Return up to ``n`` ``(text, score)`` pairs. Higher score is better.

        Does not repeat the top hypothesis to fill ``n``. If the decoder
        returns fewer unique beams, the shorter list is returned as-is.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if logits.ndim != 2:
            raise ValueError(f"Expected logits of shape (time, vocab), got {logits.shape}")

        beams = self._decoder.decode_beams(np.asarray(logits), beam_width=n)
        results: List[Tuple[str, float]] = []
        seen = set()
        for beam in beams:
            text = str(beam[0]).strip()
            score = float(beam[1]) if len(beam) > 1 else 0.0
            if text in seen:
                continue
            seen.add(text)
            results.append((text, score))
            if len(results) >= n:
                break
        return results
