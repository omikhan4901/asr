#!/usr/bin/env python3
"""
Does each metric charge a spelling or dialect difference less than a different-word error?

Sun et al. (ACL 2023) define phi-dialect robustness: a fair metric charges a change of form
(spelling or dialect) strictly less than a change of meaning. On the adjudicated Bangla word
pairs, this script scores every spelling pair and every dialect pair against every
different-word pair and reports the share of combinations where the form change costs less,
the share that ties, and the share that reverses, with a 95% bootstrap interval on the
strictly-robust share.

Cost of one (reference, hypothesis) pair:
  wer                1.0
  cer                Levenshtein.distance(ref, hyp) / len(ref)
  spelling_aware_wer 0.0 if fold(ref) == fold(hyp) else 1.0
  deep_key           0.0 if deep(ref) == deep(hyp) else 1.0
  uzzaman04_key      0.0 if uzzaman04(ref) == uzzaman04(hyp) else 1.0

  PYTHONPATH=src .venv/bin/python analysis/metric_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import fold  # noqa: E402
from spelling_keys import deep, pair_labels, uzzaman04  # noqa: E402

PAIRS = "results/labels/pair_labels.csv"
OUT = Path("results/robustness")
N_BOOT, SEED = 2000, 0
COMPARISONS = [("spelling", "different"), ("dialect", "different")]


def cer(ref: str, hyp: str) -> float:
    return Levenshtein.distance(ref, hyp) / len(ref)


def spelling_aware_wer(ref: str, hyp: str) -> float:
    return 0.0 if fold(ref) == fold(hyp) else 1.0


def deep_key(ref: str, hyp: str) -> float:
    return 0.0 if deep(ref) == deep(hyp) else 1.0


def uzzaman04_key(ref: str, hyp: str) -> float:
    return 0.0 if uzzaman04(ref) == uzzaman04(hyp) else 1.0


METRICS = {
    "wer": lambda ref, hyp: 1.0,
    "cer": cer,
    "spelling_aware_wer": spelling_aware_wer,
    "deep_key": deep_key,
    "uzzaman04_key": uzzaman04_key,
}


def shares(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Share of all (a, b) pairs with a < b, a == b, and a > b."""
    less = a[:, None] < b[None, :]
    equal = a[:, None] == b[None, :]
    return float(less.mean()), float(equal.mean()), float((~less & ~equal).mean())


def robust_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """95% CI of the strictly-robust share. a and b are resampled independently, seed 0."""
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    boots = np.empty(N_BOOT)
    batch = 250
    for start in range(0, N_BOOT, batch):
        sl = slice(start, start + batch)
        aa = a[ia[sl]]
        bb = b[ib[sl]]
        boots[sl] = (aa[:, :, None] < bb[:, None, :]).mean(axis=(1, 2))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def labelled_pairs() -> pd.DataFrame:
    pairs = pair_labels()
    pairs["ref_word"] = pairs["ref_word"].map(lambda w: normalize_bangla_text(str(w)))
    pairs["hyp_word"] = pairs["hyp_word"].map(lambda w: normalize_bangla_text(str(w)))
    keep = (pairs["ref_word"] != "") & (pairs["hyp_word"] != "") & (pairs["ref_word"] != pairs["hyp_word"])
    return pairs.loc[keep].reset_index(drop=True)


def cost_column(pairs: pd.DataFrame, fn) -> np.ndarray:
    return np.array([fn(ref, hyp) for ref, hyp in zip(pairs["ref_word"], pairs["hyp_word"])], dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with RunRecord(OUT / "run_info_metrics.json", inputs=[PAIRS]) as rec:
        pairs = labelled_pairs()
        rows = []
        for name, fn in METRICS.items():
            cost = cost_column(pairs, fn)
            for first, other in COMPARISONS:
                a = cost[pairs["label"].to_numpy() == first]
                b = cost[pairs["label"].to_numpy() == other]
                robust, tie, rev = shares(a, b)
                lo, hi = robust_ci(a, b)
                rows.append({
                    "metric": name,
                    "comparison": f"{first}_vs_{other}",
                    "n_first": int(len(a)),
                    "n_different": int(len(b)),
                    "robust": robust,
                    "robust_lo": lo,
                    "robust_hi": hi,
                    "tie": tie,
                    "reversed": rev,
                })
        out = pd.DataFrame(rows)
        path = OUT / "metric_robustness.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        rec.outputs = [path]
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
