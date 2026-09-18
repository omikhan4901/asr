"""Garbled-span detection: swappable detectors and per-span (not per-token) cap."""

from __future__ import annotations

from bangla_asr.config import SWERConfig
from bangla_asr.metrics.garbled import (
    CompositeGarbledDetector,
    ExplicitListDetector,
    WordlistOOVDetector,
    build_garbled_detector,
    is_character_implausible,
)
from bangla_asr.metrics.swer import apply_garbled_span_cap, compute_swer_result


def test_character_heuristic_flags_collapse_artifacts() -> None:
    assert is_character_implausible("ককক")
    assert is_character_implausible("ামি")  # leading matra
    assert is_character_implausible("ক্")  # trailing virama
    assert not is_character_implausible("আমি")
    assert not is_character_implausible("ইনসুলিন")


def test_explicit_detector_is_list_only() -> None:
    det = ExplicitListDetector(["ইনসলন"])
    assert det.is_garbled("ইনসলন")
    assert not det.is_garbled("ককক")


def test_wordlist_oov_needs_a_nonempty_list() -> None:
    empty = WordlistOOVDetector(wordlist=[])
    assert empty.is_garbled("anything") is False
    filled = WordlistOOVDetector(wordlist=["আমি", "গান"])
    assert filled.is_garbled("আমি") is False
    assert filled.is_garbled("ব্সক্মxyz") is True


def test_composite_default_catches_ctc_style_collapse() -> None:
    det = CompositeGarbledDetector(wordlist=["আমি", "ইনসুলিন", "গান"], threshold=2)
    assert det.is_garbled("ককক") is True
    assert det.is_garbled("আমি") is False
    assert det.is_garbled("ইনসুলিন") is False
    # Far OOV, not a near-miss of a listed word.
    assert det.is_garbled("ব্সক্মzzz") is True


def test_factory_rejects_unknown_name() -> None:
    try:
        build_garbled_detector("not-real", wordlist=["আমি"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_one_token_and_three_token_garbled_span_cost_the_same() -> None:
    """Contiguous garbled hyp tokens are one span: numerator 5.0, not 15.0."""
    ref = "ইনসুলিন"
    one = compute_swer_result("ককক", ref)
    three = compute_swer_result("ককক খখখ গগগ", ref)
    assert one.numerator == 5.0
    assert three.numerator == 5.0
    assert one.denominator == three.denominator
    assert one.swer == three.swer
    continuations = [op for op in three.operations if op.category == "garbled_span_continuation"]
    assert len(continuations) == 2
    assert all(op.weight == 0.0 for op in continuations)


def test_garbled_span_cap_helper_merges_consecutive_hyp_indices() -> None:
    from bangla_asr.common.metrics import AlignmentOp
    from bangla_asr.metrics.swer import ScoredAlignmentOp

    ops = [
        ScoredAlignmentOp(AlignmentOp("substitute", "a", "x", 0, 0), "garbled_substitution", 5.0),
        ScoredAlignmentOp(AlignmentOp("insert", None, "y", None, 1), "garbled_insertion", 5.0),
        ScoredAlignmentOp(AlignmentOp("insert", None, "z", None, 2), "garbled_insertion", 5.0),
    ]
    capped = apply_garbled_span_cap(ops, span_weight=5.0)
    assert [op.weight for op in capped] == [5.0, 0.0, 0.0]


def test_explicit_list_still_fires_on_placeholder_token() -> None:
    result = compute_swer_result("ইনসলন নাও", "ইনসুলিন নাও")
    garbled = [op for op in result.operations if op.category.startswith("garbled")]
    assert garbled
    assert result.numerator == 5.0


def test_swer_config_default_detector_is_composite() -> None:
    assert SWERConfig().garbled_detector == "composite"


if __name__ == "__main__":
    test_character_heuristic_flags_collapse_artifacts()
    test_explicit_detector_is_list_only()
    test_wordlist_oov_needs_a_nonempty_list()
    test_composite_default_catches_ctc_style_collapse()
    test_factory_rejects_unknown_name()
    test_one_token_and_three_token_garbled_span_cost_the_same()
    test_garbled_span_cap_helper_merges_consecutive_hyp_indices()
    test_explicit_list_still_fires_on_placeholder_token()
    test_swer_config_default_detector_is_composite()
    print("[PASS] garbled detector tests")
