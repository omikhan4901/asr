"""
Guarded evaluation metrics (WER, CER, pairwise loss) for ASR hypotheses.

Edit distance uses rapidfuzz's C implementation over token sequences (same
value as Wagner–Fischer). The pure-Python table below is kept only for
``align_tokens``, which needs the backtrace for error analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
import jiwer
import numpy as np
from rapidfuzz.distance import Levenshtein

from bangla_asr.common.text import normalize_bangla_text


@dataclass(frozen=True)
class AlignmentOp:
    """One step of a token-level Levenshtein alignment."""

    op: str
    ref_token: Optional[str]
    hyp_token: Optional[str]
    ref_index: Optional[int]
    hyp_index: Optional[int]


def _levenshtein_table(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> List[List[int]]:
    """Build the Wagner–Fischer distance table used by WER and SWER.

    Recurrence is identical to the previous rolling-row implementation:
    deletion, insertion, then substitution/equal with cost 0 or 1.
    """
    n_ref = len(ref_tokens)
    n_hyp = len(hyp_tokens)
    table = [[0] * (n_hyp + 1) for _ in range(n_ref + 1)]
    for j in range(n_hyp + 1):
        table[0][j] = j
    for i in range(n_ref + 1):
        table[i][0] = i
    for i, ref_tok in enumerate(ref_tokens, start=1):
        for j, hyp_tok in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            table[i][j] = min(
                table[i - 1][j] + 1,       # deletion
                table[i][j - 1] + 1,       # insertion
                table[i - 1][j - 1] + cost,  # substitution / equal
            )
    return table


def _levenshtein_distance(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> int:
    """Token-level edit distance (unit costs), computed by rapidfuzz."""
    return Levenshtein.distance(list(ref_tokens), list(hyp_tokens))


def align_tokens(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
) -> List[AlignmentOp]:
    """Backtrace the shared Levenshtein table into an edit script.

    Tie-break when several operations have the same cost (this affects SWER
    weights, not WER): prefer diagonal equal/substitute, then deletion, then
    insertion. Consecutive garbled tokens are **not** merged into one span.
    """
    table = _levenshtein_table(ref_tokens, hyp_tokens)
    i = len(ref_tokens)
    j = len(hyp_tokens)
    ops: List[AlignmentOp] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if table[i][j] == table[i - 1][j - 1] + cost:
                op = "equal" if cost == 0 else "substitute"
                ops.append(
                    AlignmentOp(
                        op=op,
                        ref_token=str(ref_tokens[i - 1]),
                        hyp_token=str(hyp_tokens[j - 1]),
                        ref_index=i - 1,
                        hyp_index=j - 1,
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append(
                AlignmentOp(
                    op="delete",
                    ref_token=str(ref_tokens[i - 1]),
                    hyp_token=None,
                    ref_index=i - 1,
                    hyp_index=None,
                )
            )
            i -= 1
            continue
        if j > 0 and table[i][j] == table[i][j - 1] + 1:
            ops.append(
                AlignmentOp(
                    op="insert",
                    ref_token=None,
                    hyp_token=str(hyp_tokens[j - 1]),
                    ref_index=None,
                    hyp_index=j - 1,
                )
            )
            j -= 1
            continue
        raise RuntimeError("Levenshtein backtrace failed; table is inconsistent with the recurrence.")
    ops.reverse()
    return ops


def word_error_rate(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """Utterance-level WER. Returns NaN when the (normalized) reference is empty."""
    if normalize:
        hyp = normalize_bangla_text(hypothesis)
        ref = normalize_bangla_text(reference)
    else:
        hyp = str(hypothesis).strip()
        ref = str(reference).strip()

    ref_tokens = ref.split()
    if not ref_tokens:
        return float("nan")
    hyp_tokens = hyp.split()
    return _levenshtein_distance(ref_tokens, hyp_tokens) / float(len(ref_tokens))


def compute_wer(
    hypotheses: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    Computes corpus-level Word Error Rate (WER).
    Safely filters out empty references to avoid jiwer zero-division errors.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"Hypothesis count ({len(hypotheses)}) does not match reference count ({len(references)})")

    if normalize:
        hyps = [normalize_bangla_text(h) for h in hypotheses]
        refs = [normalize_bangla_text(r) for r in references]
    else:
        hyps = [str(h).strip() for h in hypotheses]
        refs = [str(r).strip() for r in references]

    valid_pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]
    if not valid_pairs:
        return float("nan")

    filtered_hyps, filtered_refs = zip(*valid_pairs)
    return float(jiwer.wer(list(filtered_refs), list(filtered_hyps)))


def compute_cer(
    hypotheses: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    Computes corpus-level Character Error Rate (CER).
    """
    if len(hypotheses) != len(references):
        raise ValueError("Length mismatch between hypotheses and references")

    if normalize:
        hyps = [normalize_bangla_text(h) for h in hypotheses]
        refs = [normalize_bangla_text(r) for r in references]
    else:
        hyps = [str(h).strip() for h in hypotheses]
        refs = [str(r).strip() for r in references]

    valid_pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]
    if not valid_pairs:
        return float("nan")

    filtered_hyps, filtered_refs = zip(*valid_pairs)
    return float(jiwer.cer(list(filtered_refs), list(filtered_hyps)))


def compute_pairwise_loss(hyp: str, ref: str, empty_penalty: float = 1.0) -> float:
    """
    Computes normalized word error loss L(hyp, ref) between two candidate strings.
    If ref is empty, returns empty_penalty.
    """
    wer = word_error_rate(hyp, ref, normalize=True)
    if np.isnan(wer):
        return empty_penalty
    if not normalize_bangla_text(hyp):
        return empty_penalty
    return float(wer)
