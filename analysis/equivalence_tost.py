#!/usr/bin/env python3
"""
Is the spelling-normalization gain in keyword retrieval equivalent to zero within ±1 MAP point?

A claim that normalization adds nothing is a claim of equivalence, and a difference test cannot
show it. Two one-sided tests (TOST) can: the 90% paired-bootstrap interval of the mean gain must
lie inside the margin (Chari et al., SIGIR 2023). Per term, d = 100 * (AP with fold matching -
AP with exact matching). The margin is MARGIN = 1.0 MAP point.

  PYTHONPATH=src .venv/bin/python analysis/equivalence_tost.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402

PER_QUERY = "results/retrieval/per_query.csv"
OUT = Path("results/robustness")
SETS = ["splits_fa-eval", "fleurs-test", "cv-test", "kathbath-valid"]
SYSTEMS = ["final", "zero-shot"]
P29A_SETS = {"splits_fa-eval", "cv-test", "kathbath-valid"}
MARGIN = 1.0
N_BOOT, SEED = 2000, 0


def prediction(system: str, dataset: str) -> str:
    if system == "final" and dataset in P29A_SETS:
        return "P29a"
    if system == "zero-shot" and dataset == "splits_fa-eval":
        return "P29b"
    return ""


def interval(d: np.ndarray) -> dict:
    """Paired bootstrap of mean d over terms. One seed-0 stream per (set, system)."""
    n = len(d)
    means = d[np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    ci90_lo, ci90_hi = np.percentile(means, [5, 95])
    ci95_lo, ci95_hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_terms": n,
        "mean_d": float(d.mean()),
        "ci90_lo": float(ci90_lo),
        "ci90_hi": float(ci90_hi),
        "ci95_lo": float(ci95_lo),
        "ci95_hi": float(ci95_hi),
        "margin": MARGIN,
        "equivalent": bool(ci90_lo > -MARGIN and ci90_hi < MARGIN),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with RunRecord(OUT / "run_info_tost.json", inputs=[PER_QUERY]) as rec:
        df = pd.read_csv(PER_QUERY, encoding="utf-8-sig")
        df = df[df["task"] == "keyword"]
        rows = []
        for dataset in SETS:
            for system in SYSTEMS:
                sub = df[(df["set"] == dataset) & (df["system"] == system)
                         & df["condition"].isin(["exact", "fold"])]
                wide = sub.pivot(index="query", columns="condition", values="ap")
                d = (100 * (wide["fold"] - wide["exact"])).to_numpy(dtype=float)
                rows.append({"set": dataset, "system": system, **interval(d),
                             "prediction": prediction(system, dataset)})
        out = pd.DataFrame(rows)
        path = OUT / "tost.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        rec.outputs = [path]
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
