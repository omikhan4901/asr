"""Interface tests: Hypothesis wire format and Whisper adapter without loading weights."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bangla_asr.config import MBRConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_adapter import (
    WhisperModel,
    hypotheses_from_candidate_dicts,
)
from bangla_asr.models.base import ASRModel, Hypothesis
from bangla_asr.models.registry import build_asr_model
from bangla_asr.rescoring.mbr import MBRSelector


class _FakeGenerator:
    def __init__(self, candidates: List[Dict]) -> None:
        self.candidates = candidates
        self.last_num_candidates = None

    def generate_batch(self, audio_arrays, num_candidates=None):
        self.last_num_candidates = num_candidates
        return [list(self.candidates) for _ in audio_arrays]


def test_hypothesis_wire_format_matches_mbr() -> None:
    hyp = Hypothesis(text="আমি গান গাই", logprob=-0.15, rank=0)
    assert hyp.to_candidate_dict() == {
        "text": "আমি গান গাই",
        "logprob": -0.15,
        "rank": 0,
    }


def test_whisper_adapter_preserves_candidate_dicts() -> None:
    raw = [
        {"text": "আমি বাংলায় গান গাই", "logprob": -0.15, "rank": 0},
        {"text": "আমি বাংলায় গান খাই", "logprob": -0.35, "rank": 1},
    ]
    model = WhisperModel(
        config=WhisperConfig(num_beams=2),
        wrapper=object(),  # unused when generator is injected
        generator=_FakeGenerator(raw),
    )
    assert isinstance(model, ASRModel)
    audio = np.zeros(1600, dtype=np.float32)
    hyps = model.transcribe_nbest(audio, n=2)
    assert [h.to_candidate_dict() for h in hyps] == raw
    # n matches configured beams, so the published generate_batch path is used.
    assert model.generator.last_num_candidates is None


def test_whisper_adapter_requests_explicit_n_when_different_from_config() -> None:
    raw = [{"text": "এক", "logprob": -0.1, "rank": 0}]
    model = WhisperModel(
        config=WhisperConfig(num_beams=12),
        wrapper=object(),
        generator=_FakeGenerator(raw),
    )
    model.transcribe_nbest(np.zeros(1600, dtype=np.float32), n=1)
    assert model.generator.last_num_candidates == 1


def test_mbr_accepts_adapter_candidates() -> None:
    candidates = hypotheses_from_candidate_dicts(
        [
            {"text": "আমি বাংলায় গান গাই", "logprob": -0.15, "rank": 0},
            {"text": "আমি বাংলায় গান খাই", "logprob": -0.35, "rank": 1},
            {"text": "আমি বাংলায় গান গাই", "logprob": -0.18, "rank": 2},
        ]
    )
    chosen = MBRSelector(MBRConfig(temperature=0.5)).select(
        [h.to_candidate_dict() for h in candidates]
    )
    assert chosen["text"] == "আমি বাংলায় গান গাই"
    assert "mbr_risk" in chosen


def test_registry_rejects_unknown_family() -> None:
    try:
        build_asr_model("not-a-model")
    except ValueError as exc:
        assert "Unknown model_family" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown family")


if __name__ == "__main__":
    test_hypothesis_wire_format_matches_mbr()
    test_whisper_adapter_preserves_candidate_dicts()
    test_whisper_adapter_requests_explicit_n_when_different_from_config()
    test_mbr_accepts_adapter_candidates()
    test_registry_rejects_unknown_family()
    print("[PASS] ASR interface tests")
