"""
Phase 3 ablation: the convention rewriter with spelling/dialect rules only
(kind="variant") at the dev-chosen thresholds (min_count 2, min_p 0.5), versus
all rules, on every set. Nothing is chosen on a test set: thresholds and kind
come from the dev grid of pipeline/12_raise_ceiling.py; this only reports the
variant-only row for the same thresholds.

Usage: PYTHONPATH=src ./Scripts/python.exe analysis/phase3_rewrite_ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from rapidfuzz.distance import Levenshtein  # noqa: E402

from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.rescoring.convention import count_pairs, rewrite, select_rules  # noqa: E402

TUGP = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
SETS = ["splits_fa-dev", "splits_fa-eval", "fleurs-test", "kathbath-valid"]
OUT = Path("results/phase3/rewrite_ablation.json")


def norm(t):
    return normalize_bangla_text(t).split()


def main():
    train = f"results/zeroshot/splits_fa-train/{TUGP}/predictions.jsonl"
    with RunRecord(OUT.with_suffix(".run_info.json"), inputs=[train]) as rec:
        stats = count_pairs((norm(r["ref"]), norm(r["hyp"])) for r in read_jsonl(train))
        rules = {k: select_rules(stats, 2, 0.5, k) for k in ("all", "variant")}
        res = {"thresholds": {"min_count": 2, "min_p": 0.5}, "n_rules": {k: len(v) for k, v in rules.items()}, "sets": {}}
        for s in SETS:
            preds = read_jsonl(f"results/zeroshot/{s}/{TUGP}/predictions.jsonl")
            refs = [norm(r["ref"]) for r in preds]
            keep = [i for i, r in enumerate(refs) if r]
            groups = [str(preds[i]["group"]) for i in keep]
            words = np.array([len(refs[i]) for i in keep])
            greedy = np.array([Levenshtein.distance(refs[i], norm(preds[i]["hyp"])) for i in keep])
            row = {"greedy_wer": greedy.sum() / words.sum()}
            for k, rl in rules.items():
                e = np.array([Levenshtein.distance(refs[i], rewrite(norm(preds[i]["hyp"]), rl)) for i in keep])
                d = bootstrap_ratio(groups, e - greedy, words, n_boot=2000, seed=0)
                row[f"{k}_wer"] = e.sum() / words.sum()
                row[f"{k}_diff"], row[f"{k}_ci"] = d["value"], d["ci95"]
            res["sets"][s] = row
            print(f"{s:15s} greedy {100 * row['greedy_wer']:.2f}  all {100 * row['all_wer']:.2f} ({100 * row['all_diff']:+.2f})"
                  f"  variant-only {100 * row['variant_wer']:.2f} ({100 * row['variant_diff']:+.2f} "
                  f"[{100 * row['variant_ci'][0]:+.2f}, {100 * row['variant_ci'][1]:+.2f}])")
        atomic_write_text(OUT, json.dumps(res, indent=2))
        rec.outputs = [OUT]
    print("rules:", res["n_rules"])


if __name__ == "__main__":
    main()
