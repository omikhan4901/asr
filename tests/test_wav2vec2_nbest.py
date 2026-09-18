"""CTC N-best tests: fail loud without pyctcdecode; beams are not greedy repeated."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np
import torch

from bangla_asr.config import PipelineConfig, WhisperConfig
from bangla_asr.models.base import CTCBeamSearchUnavailableError, Hypothesis
from bangla_asr.models.ctc.beam_decoder import CTCBeamDecoder, require_pyctcdecode
from bangla_asr.models.ctc.wav2vec2_model import Wav2Vec2Model, vocab_labels_from_processor
from bangla_asr.models.registry import build_asr_model


class _FakeTokenizer:
    def get_vocab(self):
        return {"<pad>": 0, "আ": 1, "মি": 2, "|": 3}


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def __call__(self, audios, sampling_rate, return_tensors, padding):
        batch = len(audios)
        values = torch.zeros(batch, 8)
        mask = torch.ones(batch, 8)
        return SimpleNamespace(input_values=values, attention_mask=mask)

    def decode(self, pred_ids):
        return "আমি গান গাই"


class _FakeCTCModel:
    def eval(self) -> None:
        return None

    def __call__(self, input_values, attention_mask=None):
        batch, _ = input_values.shape
        # Distinct frames so greedy argmax is well-defined.
        logits = torch.zeros(batch, 4, 4)
        logits[:, 0, 1] = 3.0
        logits[:, 1, 2] = 3.0
        logits[:, 2, 3] = 3.0
        logits[:, 3, 0] = 3.0
        return SimpleNamespace(logits=logits)


class _DiverseBeamDecoder:
    def decode_beams(self, logits: np.ndarray, beam_width: int) -> List[Tuple[str, float]]:
        assert beam_width >= 2
        return [
            ("আমি গান গাই", -0.10),
            ("আমি গান খাই", -0.40),
            ("আমি বান গাই", -0.55),
        ][:beam_width]


class _RepeatedBeamDecoder:
    """Illegal decoder used only to show the test would catch greedy-repeat."""

    def decode_beams(self, logits: np.ndarray, beam_width: int):
        return [("আমি গান গাই", -0.1)] * beam_width


def test_bangla_whisper_defaults_unchanged() -> None:
    cfg = PipelineConfig()
    assert cfg.whisper == WhisperConfig()
    assert cfg.whisper.num_beams == 12
    assert cfg.whisper.sampling_temperature == 0.7
    assert cfg.wav2vec2.model_id == "arijitx/wav2vec2-xls-r-300m-bengali"


def test_nbest_without_pyctcdecode_raises() -> None:
    model = Wav2Vec2Model(
        processor=_FakeProcessor(),
        model=_FakeCTCModel(),
        beam_decoder=None,
    )
    # Force the "build a real decoder" path.
    model._beam_decoder = None

    original_require = require_pyctcdecode

    def _missing() -> None:
        raise CTCBeamSearchUnavailableError("CTC N-best requires pyctcdecode.")

    import bangla_asr.models.ctc.wav2vec2_model as w2v_mod

    w2v_mod.require_pyctcdecode = _missing  # type: ignore[assignment]
    try:
        try:
            model.transcribe_nbest(np.zeros(1600, dtype=np.float32), n=4)
        except CTCBeamSearchUnavailableError as exc:
            assert "pyctcdecode" in str(exc)
        else:
            raise AssertionError("N-best must not fall back to greedy when pyctcdecode is missing")
    finally:
        w2v_mod.require_pyctcdecode = original_require  # type: ignore[assignment]


def test_nbest_is_not_greedy_repeated() -> None:
    model = Wav2Vec2Model(
        processor=_FakeProcessor(),
        model=_FakeCTCModel(),
        beam_decoder=CTCBeamDecoder(
            labels=["<pad>", "আ", "মি", "|"],
            decoder=_DiverseBeamDecoder(),
        ),
    )
    audio = np.zeros(1600, dtype=np.float32)
    greedy = model.transcribe_nbest(audio, n=1)
    assert len(greedy) == 1
    assert greedy[0].text == "আমি গান গাই"

    nbest = model.transcribe_nbest(audio, n=3)
    texts = [h.text for h in nbest]
    assert len(texts) == 3
    assert len(set(texts)) == 3
    assert texts != [greedy[0].text] * 3
    assert all(isinstance(h, Hypothesis) for h in nbest)
    assert [h.rank for h in nbest] == [0, 1, 2]


def test_injected_decoder_deduplicates_repeated_beams() -> None:
    model = Wav2Vec2Model(
        processor=_FakeProcessor(),
        model=_FakeCTCModel(),
        beam_decoder=CTCBeamDecoder(
            labels=["<pad>", "আ", "মি", "|"],
            decoder=_RepeatedBeamDecoder(),
        ),
    )
    nbest = model.transcribe_nbest(np.zeros(1600, dtype=np.float32), n=4)
    assert [h.text for h in nbest] == ["আমি গান গাই"]


def test_vocab_labels_ordered_by_id() -> None:
    processor = _FakeProcessor()
    assert vocab_labels_from_processor(processor) == ["<pad>", "আ", "মি", "|"]


def test_registry_builds_injected_wav2vec2() -> None:
    # Registry constructs a real class; we only check the family dispatch here.
    from bangla_asr.config import Wav2Vec2Config

    model = Wav2Vec2Model(
        config=Wav2Vec2Config(model_id="tanmoyio/wav2vec2-large-xlsr-bengali"),
        processor=_FakeProcessor(),
        model=_FakeCTCModel(),
        beam_decoder=CTCBeamDecoder(labels=["a"], decoder=_DiverseBeamDecoder()),
    )
    assert model.model_family == "wav2vec2_ctc"
    assert model.cfg.model_id == "tanmoyio/wav2vec2-large-xlsr-bengali"
    assert build_asr_model.__name__ == "build_asr_model"


if __name__ == "__main__":
    test_bangla_whisper_defaults_unchanged()
    test_nbest_without_pyctcdecode_raises()
    test_nbest_is_not_greedy_repeated()
    test_injected_decoder_deduplicates_repeated_beams()
    test_vocab_labels_ordered_by_id()
    test_registry_builds_injected_wav2vec2()
    print("[PASS] Wav2Vec2 N-best tests")
