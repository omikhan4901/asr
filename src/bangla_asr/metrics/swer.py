"""
Clinical-severity-weighted error rate (SWER).

This is **not** Roy 2021 Semantic WER, not HALAS, and not "WER is Unaware"
semantic similarity. SWER multiplies standard Levenshtein edit types by a
configurable clinical-severity table.

Judgment calls (also listed in docs/STATUS.md):

* Denominator is the sum of reference-token weights. Critical tokens use
  ``critical_reference_weight`` (5.0); others use 1.0. The draft did not
  specify this mass — it is an interpretation.
* Critical deletion 4.0 vs critical substitution 5.0 vs denom 5.0 yields
  0.8 vs 1.0 for a single critical token. Left unequal on purpose; see
  ``SWERConfig`` docstring.
* Garbled detection uses a swappable ``GarbledDetector`` (default
  ``composite``). Adjacent garbled hypothesis tokens form one span and
  are charged ``garbled_span_weight`` once, not per token.
* Insertions add to the numerator only.
* Empty references have denominator 0; SWER is NaN (same as WER).
* Alignment ties: diagonal, then delete, then insert (affects SWER, not WER).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from bangla_asr.common.metrics import AlignmentOp, align_tokens
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.config import SWERConfig
from bangla_asr.metrics.garbled import GarbledDetector, build_garbled_detector, load_bangla_wordlist
from bangla_asr.metrics.severity_lexicon import SeverityLexicon, load_severity_lexicon

GARBLED_CATEGORIES = frozenset({"garbled_substitution", "garbled_insertion"})


@dataclass(frozen=True)
class ScoredAlignmentOp:
    """An alignment step plus the SWER weight charged to the numerator."""

    alignment: AlignmentOp
    category: str
    weight: float


@dataclass(frozen=True)
class SWERResult:
    """Utterance-level SWER with the scored alignment used to compute it."""

    swer: float
    numerator: float
    denominator: float
    operations: List[ScoredAlignmentOp]


def _tokenize(text: str) -> List[str]:
    return normalize_bangla_text(text).split()


def is_empty_reference(reference: str) -> bool:
    """True when SWER's denominator would be 0 (no tokens after normalization)."""
    return len(_tokenize(reference)) == 0


def _reference_token_weight(is_critical: bool, config: SWERConfig) -> float:
    if is_critical:
        return config.critical_reference_weight
    return config.default_reference_weight


def _lexicon_real_words(lexicon: SeverityLexicon) -> Set[str]:
    words = set(lexicon.known_words) | set(lexicon.critical_tokens) | set(lexicon.filler_tokens)
    for phrase in lexicon.critical_phrases:
        words.update(phrase)
    return words


def detector_from_config(
    config: SWERConfig,
    lexicon: SeverityLexicon,
) -> GarbledDetector:
    """Build the configured garbled detector, unioning lexicon terms into the wordlist."""
    wordlist = load_bangla_wordlist(config.garbled_wordlist_path)
    wordlist |= _lexicon_real_words(lexicon)
    return build_garbled_detector(
        config.garbled_detector,
        wordlist=wordlist,
        garbled_tokens=lexicon.garbled_tokens,
        threshold=config.garbled_edit_threshold,
        treat_oov_as_garbled=config.treat_oov_as_garbled,
    )


def classify_edit(
    op: AlignmentOp,
    *,
    lexicon: SeverityLexicon,
    config: SWERConfig,
    critical_ref: Sequence[bool],
    hyp_tokens: Sequence[str],
    detector: GarbledDetector,
) -> ScoredAlignmentOp:
    """Map one Levenshtein op onto the draft SWER category table."""
    if op.op == "equal":
        return ScoredAlignmentOp(alignment=op, category="equal", weight=0.0)

    if op.op == "substitute":
        ref_tok = op.ref_token or ""
        hyp_tok = op.hyp_token or ""
        if detector.is_garbled(hyp_tok):
            return ScoredAlignmentOp(
                alignment=op,
                category="garbled_substitution",
                weight=config.garbled_span_weight,
            )
        if lexicon.is_critical_substitution(ref_tok, hyp_tok):
            return ScoredAlignmentOp(
                alignment=op,
                category="critical_substitution",
                weight=config.critical_substitution_weight,
            )
        return ScoredAlignmentOp(
            alignment=op,
            category="minor_substitution",
            weight=config.minor_substitution_weight,
        )

    if op.op == "delete":
        ref_idx = op.ref_index
        is_critical = bool(ref_idx is not None and ref_idx < len(critical_ref) and critical_ref[ref_idx])
        if is_critical:
            return ScoredAlignmentOp(
                alignment=op,
                category="critical_deletion",
                weight=config.critical_deletion_weight,
            )
        return ScoredAlignmentOp(
            alignment=op,
            category="noncritical_deletion",
            weight=config.noncritical_deletion_weight,
        )

    if op.op == "insert":
        hyp_tok = op.hyp_token or ""
        # Repeated = consecutive duplicate of a neighbor in the hypothesis.
        # Levenshtein often inserts the first copy, so check both sides.
        hyp_idx = op.hyp_index
        repeated = False
        if hyp_idx is not None and 0 <= hyp_idx < len(hyp_tokens):
            tok = hyp_tokens[hyp_idx]
            if hyp_idx > 0 and hyp_tokens[hyp_idx - 1] == tok:
                repeated = True
            if hyp_idx + 1 < len(hyp_tokens) and hyp_tokens[hyp_idx + 1] == tok:
                repeated = True
        if lexicon.is_filler(hyp_tok) or repeated:
            return ScoredAlignmentOp(
                alignment=op,
                category="filler_insertion",
                weight=config.filler_insertion_weight,
            )
        if detector.is_garbled(hyp_tok):
            return ScoredAlignmentOp(
                alignment=op,
                category="garbled_insertion",
                weight=config.garbled_span_weight,
            )
        return ScoredAlignmentOp(
            alignment=op,
            category="hallucinated_insertion",
            weight=config.hallucinated_insertion_weight,
        )

    raise ValueError(f"Unknown alignment op: {op.op}")


def apply_garbled_span_cap(
    scored: Sequence[ScoredAlignmentOp],
    span_weight: float,
) -> List[ScoredAlignmentOp]:
    """Charge ``span_weight`` once per contiguous run of garbled hyp tokens.

    Contiguous means consecutive ``hyp_index`` values. Continuations keep the
    alignment row but contribute 0 to the numerator.
    """
    result = list(scored)
    garbled = [
        i
        for i, item in enumerate(result)
        if item.category in GARBLED_CATEGORIES and item.alignment.hyp_index is not None
    ]
    garbled.sort(key=lambda i: int(result[i].alignment.hyp_index))

    groups: List[List[int]] = []
    current: List[int] = []
    last_hyp: Optional[int] = None
    for idx in garbled:
        hyp_i = int(result[idx].alignment.hyp_index)
        if last_hyp is None or hyp_i == last_hyp + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
        last_hyp = hyp_i
    if current:
        groups.append(current)

    for group in groups:
        for offset, idx in enumerate(group):
            old = result[idx]
            if offset == 0:
                result[idx] = ScoredAlignmentOp(old.alignment, old.category, float(span_weight))
            else:
                result[idx] = ScoredAlignmentOp(old.alignment, "garbled_span_continuation", 0.0)
    return result


def compute_swer_result(
    hypothesis: str,
    reference: str,
    *,
    lexicon: Optional[SeverityLexicon] = None,
    config: Optional[SWERConfig] = None,
    detector: Optional[GarbledDetector] = None,
) -> SWERResult:
    """Compute utterance SWER and the scored alignment used to get it."""
    config = config or SWERConfig()
    lexicon = lexicon or load_severity_lexicon(config.lexicon_path)
    if detector is None:
        detector = detector_from_config(config, lexicon)
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
    critical_ref = lexicon.mark_critical_reference_tokens(ref_tokens)
    denominator = float(sum(_reference_token_weight(flag, config) for flag in critical_ref))

    ops = align_tokens(ref_tokens, hyp_tokens)
    scored = [
        classify_edit(
            op,
            lexicon=lexicon,
            config=config,
            critical_ref=critical_ref,
            hyp_tokens=hyp_tokens,
            detector=detector,
        )
        for op in ops
    ]
    scored = apply_garbled_span_cap(scored, config.garbled_span_weight)
    numerator = float(sum(item.weight for item in scored))

    if denominator <= 0.0:
        return SWERResult(swer=float("nan"), numerator=numerator, denominator=denominator, operations=scored)
    return SWERResult(
        swer=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        operations=scored,
    )


def compute_swer(
    hypothesis: str,
    reference: str,
    *,
    lexicon: Optional[SeverityLexicon] = None,
    config: Optional[SWERConfig] = None,
    detector: Optional[GarbledDetector] = None,
) -> float:
    """Utterance-level SWER. NaN when the weighted reference length is 0."""
    return compute_swer_result(
        hypothesis,
        reference,
        lexicon=lexicon,
        config=config,
        detector=detector,
    ).swer


def corpus_swer(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    lexicon: Optional[SeverityLexicon] = None,
    config: Optional[SWERConfig] = None,
    detector: Optional[GarbledDetector] = None,
) -> float:
    """Micro-averaged SWER: sum of numerators over sum of denominators.

    Utterances with an empty weighted reference are skipped, matching
    ``compute_wer``'s empty-reference filter.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Hypothesis count ({len(hypotheses)}) does not match reference count ({len(references)})"
        )
    config = config or SWERConfig()
    lexicon = lexicon or load_severity_lexicon(config.lexicon_path)
    if detector is None:
        detector = detector_from_config(config, lexicon)
    num = 0.0
    den = 0.0
    for hyp, ref in zip(hypotheses, references):
        result = compute_swer_result(hyp, ref, lexicon=lexicon, config=config, detector=detector)
        if result.denominator <= 0.0:
            continue
        num += result.numerator
        den += result.denominator
    if den <= 0.0:
        return float("nan")
    return num / den
