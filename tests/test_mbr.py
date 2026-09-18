"""MBR selector tests extracted from the smoke suite so the baseline stays locked."""

from __future__ import annotations

import numpy as np

from bangla_asr.config import MBRConfig
from bangla_asr.rescoring.mbr import MBRSelector, run_mbr_rescoring, softmax_with_temperature


def test_mbr_selects_consensus_hypothesis() -> None:
    candidates = [
        {"text": "আমি বাংলায় গান গাই", "logprob": -0.15, "rank": 0},
        {"text": "আমি বাংলায় গান খাই", "logprob": -0.35, "rank": 1},
        {"text": "আমি বাংলায় গান গাই", "logprob": -0.18, "rank": 2},
    ]
    chosen = MBRSelector(MBRConfig(temperature=0.5)).select(candidates)
    assert chosen["text"] == "আমি বাংলায় গান গাই"
    assert "mbr_risk" in chosen


def test_softmax_maps_non_finite_scores_to_uniform() -> None:
    weights = softmax_with_temperature(np.array([np.nan, np.inf]))
    assert np.allclose(weights, [0.5, 0.5])


def test_run_mbr_skips_empty_candidate_lists() -> None:
    results = run_mbr_rescoring(
        [
            {"utt_id": "valid", "ground_truth": "reference", "candidates": [{"text": "reference", "rank": 0}]},
            {"utt_id": "empty", "ground_truth": "ignored", "candidates": []},
        ]
    )
    assert results["n_utterances"] == 1
    assert results["temperature"] == 0.5


if __name__ == "__main__":
    test_mbr_selects_consensus_hypothesis()
    test_softmax_maps_non_finite_scores_to_uniform()
    test_run_mbr_skips_empty_candidate_lists()
    print("[PASS] MBR tests")
