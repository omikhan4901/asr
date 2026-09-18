"""
Baseline (1-best) and Oracle ceiling evaluator for quantifying headroom.
"""

from typing import Dict, List, Tuple
import numpy as np

from bangla_asr.common.metrics import compute_wer, compute_cer


def evaluate_1best_baseline(records: List[Dict]) -> Dict:
    """Evaluates the top-ranked (rank-0) beam hypothesis per utterance."""
    hypotheses = []
    references = []

    for rec in records:
        candidates = rec.get("candidates", [])
        if not candidates:
            continue
        top_cand = min(candidates, key=lambda c: c.get("rank", 0))
        hypotheses.append(top_cand.get("text", ""))
        references.append(rec.get("ground_truth", ""))

    wer = compute_wer(hypotheses, references)
    cer = compute_cer(hypotheses, references)
    return {
        "method": "1-Best Baseline",
        "wer": wer,
        "cer": cer,
        "n_utterances": len(hypotheses),
    }


def evaluate_oracle_ceiling(records: List[Dict]) -> Dict:
    """
    Theoretical upper bound: for each utterance, selects the candidate
    with the lowest individual WER against ground truth.
    """
    hypotheses = []
    references = []

    for rec in records:
        candidates = rec.get("candidates", [])
        ref = rec.get("ground_truth", "")
        if not candidates:
            continue

        best_cand_text = candidates[0].get("text", "")
        best_wer = float("inf")

        for cand in candidates:
            txt = cand.get("text", "")
            w = compute_wer([txt], [ref])
            if not np.isnan(w) and w < best_wer:
                best_wer = w
                best_cand_text = txt

        hypotheses.append(best_cand_text)
        references.append(ref)

    wer = compute_wer(hypotheses, references)
    cer = compute_cer(hypotheses, references)
    return {
        "method": "Oracle Ceiling",
        "wer": wer,
        "cer": cer,
        "n_utterances": len(hypotheses),
    }


def evaluate_oracle_headroom(records: List[Dict]) -> Dict:
    """Computes both 1-best and Oracle ceiling, reporting available headroom."""
    baseline = evaluate_1best_baseline(records)
    oracle = evaluate_oracle_ceiling(records)

    gap = (baseline["wer"] - oracle["wer"]) * 100.0
    return {
        "1best_baseline": baseline,
        "oracle_ceiling": oracle,
        "oracle_gap_wer_points": round(gap, 2),
    }
