"""
Hand-labeled SWER rank-order sanity check.

This is a scaffold. It does NOT assert exact SWER values. Once you fill
``tests/fixtures/swer_sanity_examples.json`` from real Whisper / Wav2Vec2
outputs and set ``ready`` to true, the test checks that computed SWER's
*relative ordering* matches your manual severity labels
(clinically_dangerous > moderate > cosmetic).

Until then the test is skipped with a loud message. That is intentional:
do not treat compare_models.py SWER columns as validated until this passes
on your own labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bangla_asr.metrics.swer import compute_swer

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "swer_sanity_examples.json"

_SEVERITY_RANK = {
    "cosmetic": 0,
    "harmless": 0,
    "minor": 0,
    "moderate": 1,
    "clinically_dangerous": 2,
    "dangerous": 2,
    "critical": 2,
}

_SKIP_MESSAGE = (
    "SWER sanity fixture is a placeholder. Fill tests/fixtures/swer_sanity_examples.json "
    "with real (asr_hypothesis, ground_truth, manual_severity) triples from Whisper and "
    "Wav2Vec2, set ready=true, then re-run. Until that happens, do not trust SWER columns "
    "in compare_models.py as clinically validated."
)


def _load_fixture() -> Dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _usable_examples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    usable = []
    for row in data.get("examples") or []:
        hyp = str(row.get("asr_hypothesis") or "").strip()
        ref = str(row.get("ground_truth") or "").strip()
        label = str(row.get("manual_severity") or "").strip().lower()
        if hyp and ref and label in _SEVERITY_RANK:
            usable.append(row)
    return usable


def _skip(message: str) -> None:
    try:
        import pytest
    except ImportError:
        print(f"[SKIP] {message}")
        return
    pytest.skip(message)


def test_swer_rank_order_matches_manual_judgment() -> None:
    data = _load_fixture()
    usable = _usable_examples(data)
    if not data.get("ready") or len(usable) < 2:
        _skip(_SKIP_MESSAGE)
        return

    scored: List[Tuple[str, int, float]] = []
    for row in usable:
        swer = compute_swer(row["asr_hypothesis"], row["ground_truth"])
        scored.append((row["id"], _SEVERITY_RANK[row["manual_severity"].strip().lower()], swer))

    inversions = []
    for i, (id_a, rank_a, swer_a) in enumerate(scored):
        for id_b, rank_b, swer_b in scored[i + 1 :]:
            if rank_a == rank_b:
                continue
            if rank_a > rank_b and swer_a <= swer_b:
                inversions.append((id_a, swer_a, id_b, swer_b))
            if rank_b > rank_a and swer_b <= swer_a:
                inversions.append((id_b, swer_b, id_a, swer_a))

    assert not inversions, (
        "SWER rank order disagrees with manual severity. "
        "Each pair is (higher_manual, swer, lower_manual, swer): "
        f"{inversions}"
    )


if __name__ == "__main__":
    test_swer_rank_order_matches_manual_judgment()
    print("[PASS or SKIP] SWER sanity harness")
