"""
Factory for acoustic models so pipeline scripts depend on ``ASRModel``, not a family.
"""

from __future__ import annotations

from typing import Any, Optional

from bangla_asr.models.base import ASRModel


def build_asr_model(
    model_family: str,
    config: Any = None,
    device: Optional[str] = None,
) -> ASRModel:
    """Construct an ``ASRModel`` for ``whisper`` or ``wav2vec2_ctc``.

    Config objects are family-specific dataclasses. Passing ``None`` uses the
    family default, which for Whisper is the published baseline configuration.
    """
    family = (model_family or "").strip().lower()
    if family in {"whisper", "autoregressive"}:
        from bangla_asr.config import WhisperConfig
        from bangla_asr.models.autoregressive.whisper_adapter import WhisperModel

        whisper_cfg = config if config is not None else WhisperConfig()
        return WhisperModel(whisper_cfg, device=device)

    if family in {"wav2vec2", "wav2vec2_ctc", "ctc"}:
        from bangla_asr.config import Wav2Vec2Config
        from bangla_asr.models.ctc.wav2vec2_model import Wav2Vec2Model

        w2v_cfg = config if config is not None else Wav2Vec2Config()
        return Wav2Vec2Model(w2v_cfg, device=device)

    raise ValueError(
        f"Unknown model_family={model_family!r}. Expected 'whisper' or 'wav2vec2_ctc'."
    )
