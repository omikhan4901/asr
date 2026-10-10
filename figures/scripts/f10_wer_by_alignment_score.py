"""
F10: WER by forced-aligner confidence (mean word posterior of the clip) on DL Sprint eval, for zero-shot
tugstugi and M1 full, with 95% cluster-bootstrap CIs; bar labels = number of clips. Clips below 0.5
were filtered out by the aligner (never by an evaluated model), so the lowest bin starts at 0.5.

Inputs: data/splits_fa/eval.csv, results/modal/{zeroshot,finetuned}/splits_fa-eval/<run>/predictions.jsonl
Run:    PYTHONPATH=src ./Scripts/python.exe figures/scripts/f10_wer_by_alignment_score.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from style import COL_W, OKABE, ROOT, plt, save  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio, edit_counts  # noqa: E402

RUNS = {"zero-shot": ("zeroshot", "tugstugi_a100", OKABE["grey"]), "M1 full": ("finetuned", "m1_full", OKABE["blue"])}
BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
man = pd.read_csv(ROOT / "data/splits_fa/eval.csv", encoding="utf-8-sig").set_index("utt_id")

rows = []
for label, (kind, run, _) in RUNS.items():
    preds = [json.loads(l) for l in open(ROOT / "results/modal" / kind / "splits_fa-eval" / run / "predictions.jsonl",
                                          encoding="utf-8")]
    df = pd.DataFrame([{**edit_counts(p["hyp"], p["ref"]), "group": p["group"], "score": man.loc[p["utt_id"], "mean_score"]}
                       for p in preds])
    df["bin"] = pd.cut(df.score, BINS, include_lowest=True)
    for b, g in df.groupby("bin", observed=True):
        r = bootstrap_ratio(list(g.group), g.word_edits, g.ref_words, n_boot=1000)
        rows.append({"model": label, "bin": f"{b.left:.1f}-{b.right:.1f}", "n": len(g), "wer": 100 * r["value"],
                     "lo": 100 * r["ci95"][0], "hi": 100 * r["ci95"][1]})
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(COL_W, 2.2))
labels = list(dict.fromkeys(d.bin))
for i, (label, (_, _, color)) in enumerate(RUNS.items()):
    g = d[d.model == label].set_index("bin").reindex(labels)
    x = np.arange(len(labels)) + (i - 0.5) * 0.38
    ax.bar(x, g.wer, 0.36, color=color, label=label)
    ax.errorbar(x, g.wer, yerr=[g.wer - g.lo, g.hi - g.wer], fmt="none", ecolor="black", lw=0.6, capsize=1.2)
    if i == 0:
        for xi, n, w in zip(x, g.n, g.hi):
            ax.text(xi + 0.19, w + 1, f"n={n}", ha="center", fontsize=6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("aligner confidence of the clip (mean word posterior)")
ax.set_ylabel("WER (%)")
ax.legend(frameon=False)
save(fig, "f10_wer_by_alignment_score", d)
