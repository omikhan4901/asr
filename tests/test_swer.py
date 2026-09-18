"""SWER alignment correctness, category weights, and placeholder-lexicon edge cases."""

from __future__ import annotations

import math

from bangla_asr.common.metrics import _levenshtein_distance, align_tokens, word_error_rate
from bangla_asr.config import PipelineConfig, WhisperConfig
from bangla_asr.metrics.severity_lexicon import PLACEHOLDER_LEXICON, load_severity_lexicon
from bangla_asr.metrics.swer import compute_swer, compute_swer_result, corpus_swer


def test_alignment_distance_matches_wer_backbone() -> None:
    pairs = [
        (["আমি", "গান", "গাই"], ["আমি", "গান", "গাই"]),
        (["আমি", "গান", "গাই"], ["আমি", "বই", "পড়ি"]),
        (["ইনসুলিন", "নাও"], ["ইনসুলিন"]),
        (["গান"], ["আমি", "গান", "গাই"]),
        ([], ["এক"]),
        (["এক"], []),
    ]
    for ref, hyp in pairs:
        ops = align_tokens(ref, hyp)
        edits = sum(1 for op in ops if op.op != "equal")
        assert edits == _levenshtein_distance(ref, hyp)


def test_perfect_match_is_zero() -> None:
    assert compute_swer("আমি গান গাই", "আমি গান গাই") == 0.0


def test_minor_substitution_weight() -> None:
    # three non-critical tokens, one substitution -> 1 / 3
    result = compute_swer_result("আমি বই গাই", "আমি গান গাই")
    assert result.denominator == 3.0
    assert result.numerator == 1.0
    assert result.swer == 1.0 / 3.0
    categories = [op.category for op in result.operations if op.category != "equal"]
    assert categories == ["minor_substitution"]


def test_critical_substitution_weight() -> None:
    result = compute_swer_result("নিম্নরক্তচাপ", "উচ্চরক্তচাপ")
    assert result.numerator == 5.0
    assert result.denominator == 5.0
    assert result.swer == 1.0
    assert result.operations[0].category == "critical_substitution"


def test_garbled_substitution_weight() -> None:
    result = compute_swer_result("ইনসলন নাও", "ইনসুলিন নাও")
    garbled = [op for op in result.operations if op.category == "garbled_substitution"]
    assert len(garbled) == 1
    assert garbled[0].weight == 5.0
    # ইনসুলিন is critical (denom 5) + নাও non-critical (denom 1)
    assert result.denominator == 6.0
    assert result.numerator == 5.0


def test_critical_and_noncritical_deletion() -> None:
    critical = compute_swer_result("নাও", "ইনসুলিন নাও")
    assert any(op.category == "critical_deletion" for op in critical.operations)
    assert critical.numerator == 4.0
    assert critical.denominator == 6.0

    minor = compute_swer_result("ইনসুলিন", "ইনসুলিন নাও")
    assert any(op.category == "noncritical_deletion" for op in minor.operations)
    assert minor.numerator == 1.0


def test_insertion_categories() -> None:
    halluc = compute_swer_result("আমি অতিরিক্ত গান গাই", "আমি গান গাই")
    assert any(op.category == "hallucinated_insertion" for op in halluc.operations)
    assert halluc.numerator == 3.0
    assert halluc.denominator == 3.0

    filler = compute_swer_result("আমি মানে গান গাই", "আমি গান গাই")
    assert any(op.category == "filler_insertion" for op in filler.operations)
    assert filler.numerator == 1.0

    repeated = compute_swer_result("আমি গান গান গাই", "আমি গান গাই")
    assert any(op.category == "filler_insertion" for op in repeated.operations)
    assert repeated.numerator == 1.0


def test_empty_reference_is_nan() -> None:
    value = compute_swer("আমি", "")
    assert math.isnan(value)


def test_placeholder_lexicon_is_marked() -> None:
    lexicon = load_severity_lexicon()
    assert lexicon.placeholder or lexicon.source.endswith("placeholder_bangla.yaml")
    assert PLACEHOLDER_LEXICON["placeholder"] is True
    assert lexicon.language == "bn"


def test_corpus_micro_average() -> None:
    value = corpus_swer(
        ["নিম্নরক্তচাপ", "আমি গান গাই"],
        ["উচ্চরক্তচাপ", "আমি গান গাই"],
    )
    # 5/5 + 0/3  -> 5 / 8
    assert abs(value - (5.0 / 8.0)) < 1e-9


def test_swer_not_equal_to_unweighted_wer_on_critical_sub() -> None:
    hyp, ref = "নিম্নরক্তচাপ", "উচ্চরক্তচাপ"
    assert word_error_rate(hyp, ref) == 1.0
    assert compute_swer(hyp, ref) == 1.0
    # Same ratio here because critical_reference_weight == critical_substitution_weight.
    # A mixed sentence diverges:
    hyp2, ref2 = "নিম্নরক্তচাপ আছে", "উচ্চরক্তচাপ আছে"
    assert word_error_rate(hyp2, ref2) == 0.5
    assert compute_swer(hyp2, ref2) == 5.0 / 6.0


def test_whisper_baseline_defaults_still_untouched() -> None:
    cfg = PipelineConfig()
    assert cfg.whisper == WhisperConfig()
    assert cfg.whisper.num_beams == 12
    assert cfg.mbr.temperature == 0.5


if __name__ == "__main__":
    test_alignment_distance_matches_wer_backbone()
    test_perfect_match_is_zero()
    test_minor_substitution_weight()
    test_critical_substitution_weight()
    test_garbled_substitution_weight()
    test_critical_and_noncritical_deletion()
    test_insertion_categories()
    test_empty_reference_is_nan()
    test_placeholder_lexicon_is_marked()
    test_corpus_micro_average()
    test_swer_not_equal_to_unweighted_wer_on_critical_sub()
    test_whisper_baseline_defaults_still_untouched()
    print("[PASS] SWER tests")
