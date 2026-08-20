"""
MBR (Minimum Bayes Risk) consensus rescoring.

FIX vs. the previous version: this now actually implements the formula
stated in the paper (Section III-D2):
    hMBR = argmin_i  sum_j P(hj | X) * L(hi, hj)

The previous version averaged L(hi, hj) UNIFORMLY across all other
candidates (implicitly P(hj) = 1/N for every j), which is a legitimate
consensus method but not the probability-weighted one written in the
methodology. This version weights each candidate's contribution to the
risk by its own acoustic-model logprob (softmax-normalized across the
N-best pool for that utterance), so a high-confidence candidate that
several others agree with pulls more weight than a low-confidence outlier.

Same CLI/output contract as before — drop-in replacement.
"""

import json
import argparse
from pathlib import Path

import jiwer
import numpy as np

from evaluation import compute_wer, compute_cer, load_nbest_records, normalize_bangla_text


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def mbr_select(candidates: list[dict]) -> dict:
    """
    Probability-weighted MBR: picks the candidate minimizing expected
    word-level error against the rest of the pool, weighted by each other
    candidate's own acoustic sequence logprob (softmax-normalized).
    """
    texts = [normalize_bangla_text(c["text"]) for c in candidates]
    logprobs = np.array([c["logprob"] for c in candidates])
    weights = softmax(logprobs)  # P(hj | X) per candidate, sums to 1

    n = len(texts)
    risks = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                loss = jiwer.wer(texts[j], texts[i])  # j = reference, i = hypothesis
            except Exception:
                loss = 1.0  # empty/degenerate candidate — penalize, don't crash
            risks[i] += weights[j] * loss

    best_idx = int(np.argmin(risks))
    return candidates[best_idx]


def run_mbr_eval(nbest_jsonl: str, output_dir: str):
    records = load_nbest_records(nbest_jsonl)
    print(f"Loaded {len(records)} utterances.")

    hyps, refs = [], []
    for rec in records:
        selected = mbr_select(rec["candidates"])
        hyps.append(selected["text"])
        refs.append(rec["ground_truth"])

    wer = compute_wer(hyps, refs)
    cer = compute_cer(hyps, refs)

    print(f"MBR WER: {wer*100:.2f}%")
    print(f"MBR CER: {cer*100:.2f}%")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mbr_results.json", "w", encoding="utf-8") as f:
        json.dump({"wer": wer, "cer": cer, "n_utterances": len(records)}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out / 'mbr_results.json'}")
    return {"wer": wer, "cer": cer}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nbest_jsonl", default="eval_nbest_candidates.jsonl")
    p.add_argument("--output_dir", default="data/processed/results/mbr")
    args = p.parse_args()
    run_mbr_eval(args.nbest_jsonl, args.output_dir)