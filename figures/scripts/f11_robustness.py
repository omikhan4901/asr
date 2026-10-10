"""
F11: robustness on DL Sprint eval: WER against (a) clip length and (b) speaking rate (reference words per
second; quintile bins), for zero-shot tugstugi and M1 full, with 95% cluster-bootstrap CIs.

Inputs: data/splits_fa/eval.csv, results/modal/{zeroshot,finetuned}/splits_fa-eval/<run>/predictions.jsonl
Run:    PYTHONPATH=src ./Scripts/python.exe figures/scripts/f11_robustness.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio, edit_counts  # noqa: E402

RUNS = {"zero-shot": ("zeroshot", "tugstugi_a100", OKABE["grey"]), "M1 full": ("finetuned", "m1_full", OKABE["blue"])}
man = pd.read_csv(ROOT / "data/splits_fa/eval.csv", encoding="utf-8-sig").set_index("utt_id")

frames = []
for label, (kind, run, _) in RUNS.items():
    preds = [json.loads(l) for l in open(ROOT / "results/modal" / kind / "splits_fa-eval" / run / "predictions.jsonl",
                                          encoding="utf-8")]
    df = pd.DataFrame([{**edit_counts(p["hyp"], p["ref"]), "group": p["group"],
                        "duration": man.loc[p["utt_id"], "duration"]} for p in preds])
    df["rate"] = df.ref_words / df.duration
    df["model"] = label
    frames.append(df)
d = pd.concat(frames)
d["len_bin"] = pd.cut(d.duration, [0, 5, 10, 15, 20, 26], labels=["<5", "5-10", "10-15", "15-20", "20-25"])
d["rate_bin"] = pd.qcut(d.rate, 5)

rows = []
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2), sharey=True)
for ax, col, xlabel in [(axes[0], "len_bin", "clip length (s)"), (axes[1], "rate_bin", "speaking rate (words/s, quintiles)")]:
    for label, (_, _, color) in RUNS.items():
        pts = []
        for b, g in d[d.model == label].groupby(col, observed=True):
            r = bootstrap_ratio(list(g.group), g.word_edits, g.ref_words, n_boot=1000)
            name = str(b) if col == "len_bin" else f"{b.left:.1f}-{b.right:.1f}"
            pts.append((name, 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1], len(g)))
            rows.append({"panel": col, "model": label, "bin": name, "wer": pts[-1][1], "lo": pts[-1][2],
                         "hi": pts[-1][3], "n": len(g)})
        x = range(len(pts))
        ax.errorbar(list(x), [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt="-o", ms=3, lw=1, capsize=1.5, color=color, label=label)
        ax.set_xticks(list(x))
        ax.set_xticklabels([p[0] for p in pts], fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, None)
axes[0].set_ylabel("WER (%) on DL Sprint eval")
axes[0].legend(frameon=False)
save(fig, "f11_robustness", pd.DataFrame(rows))
