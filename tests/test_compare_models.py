"""Side-by-side eval uses rank-0 for Whisper and greedy_text for CTC."""

from __future__ import annotations

from bangla_asr.eval.compare_models import compare_acoustic_tracks, evaluate_track


def _whisper_records():
    return [
        {
            "utt_id": "w1",
            "ground_truth": "আমি গান গাই",
            "model_family": "whisper",
            "candidates": [
                {"text": "আমি গান গাই", "logprob": -0.1, "rank": 0},
                {"text": "আমি গান খাই", "logprob": -0.4, "rank": 1},
            ],
        },
        {
            "utt_id": "w2",
            "ground_truth": "তুমি বই পড়ো",
            "model_family": "whisper",
            "candidates": [
                {"text": "তুমি বই খাও", "logprob": -0.2, "rank": 0},
                {"text": "তুমি বই পড়ো", "logprob": -0.3, "rank": 1},
            ],
        },
    ]


def _ctc_records():
    return [
        {
            "utt_id": "c1",
            "ground_truth": "আমি গান গাই",
            "model_family": "wav2vec2_ctc",
            "greedy_text": "আমি গান গাই",
            "candidates": [
                {"text": "আমি গান খাই", "logprob": -0.05, "rank": 0},
                {"text": "আমি গান গাই", "logprob": -0.20, "rank": 1},
            ],
        }
    ]


def test_whisper_one_best_is_rank_zero() -> None:
    metrics = evaluate_track(_whisper_records(), "whisper")
    # utterance 1 exact, utterance 2 2/3 substitutions -> corpus WER in (0, 1)
    assert metrics["n_utterances"] == 2
    assert 0.0 < metrics["one_best_wer"] < 1.0
    assert metrics["oracle_wer"] == 0.0
    assert metrics["mbr_wer"] <= metrics["one_best_wer"]


def test_ctc_one_best_uses_greedy_text_not_beam_rank0() -> None:
    metrics = evaluate_track(_ctc_records(), "wav2vec2")
    assert metrics["one_best_wer"] == 0.0
    # beam rank-0 is wrong; if we had used it, 1-best WER would be 1/3
    assert metrics["oracle_wer"] == 0.0


def test_compare_writes_both_tracks() -> None:
    summary = compare_acoustic_tracks(
        {"whisper": _whisper_records(), "wav2vec2": _ctc_records()}
    )
    names = [row["track"] for row in summary["tracks"]]
    assert names == ["whisper", "wav2vec2"]


if __name__ == "__main__":
    test_whisper_one_best_is_rank_zero()
    test_ctc_one_best_uses_greedy_text_not_beam_rank0()
    test_compare_writes_both_tracks()
    print("[PASS] compare_models tests")
