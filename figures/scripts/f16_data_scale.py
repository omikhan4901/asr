"""
F16: WER against training hours with the same recipe (full fine-tuning, 2 epochs, 1e-5): M5 (10 h from each of four
corpora, 40 h), M5-full (every hour of the four, 156 h), M5-300 (plus 144 h of OpenSLR 53 read speech, ~300 h), with
the in-domain M1 (60 h of DL Sprint) as a separate marker. One line per test set, 95% cluster-bootstrap CIs.

Inputs: results/kaggle/finetuned/<set>/{m5_merged,m5_full,m5_300}_t4/metrics.json,
        results/modal/finetuned/<set>/m1_full/metrics.json (A100; T4 = A100 within 0.01 points, R46)
Run:    ./Scripts/python.exe figures/scripts/f16_data_scale.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, COL_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-eval": ("DL Sprint eval", OKABE["blue"]), "fleurs-test": ("FLEURS", OKABE["orange"]),
        "cv-test": ("Common Voice", OKABE["purple"]), "kathbath-valid": ("Kathbath valid", OKABE["green"])}
RUNS = [("M5", 40, "results/kaggle/finetuned/{s}/m5_merged_t4"), ("M5-full", 156, "results/kaggle/finetuned/{s}/m5_full_t4"),
        ("M5-300", 300, "results/kaggle/finetuned/{s}/m5_300_t4")]
M1 = ("M1 (DL Sprint only)", 60, "results/modal/finetuned/{s}/m1_full")


def metrics(p):
    m = json.loads((ROOT / p / "metrics.json").read_text(encoding="utf-8"))
    return 100 * m["wer"], 100 * m["wer_ci95"][0], 100 * m["wer_ci95"][1]


rows = []
for s, (name, _) in SETS.items():
    for run, hours, p in RUNS + [M1]:
        w, lo, hi = metrics(p.format(s=s))
        rows.append({"set": name, "run": run, "hours": hours, "wer": w, "lo": lo, "hi": hi})
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(COL_W, 2.4))
for s, (name, col) in SETS.items():
    g = d[(d.set == name) & (d.run != M1[0])]
    ax.errorbar(g.hours, g.wer, yerr=[g.wer - g.lo, g.hi - g.wer], fmt="-o", ms=3.5, lw=1, capsize=1.5, color=col, label=name)
    m = d[(d.set == name) & (d.run == M1[0])].iloc[0]
    ax.errorbar(m.hours, m.wer, yerr=[[m.wer - m.lo], [m.hi - m.wer]], fmt="s", ms=3.5, lw=0.8, capsize=1.5,
                color=col, mfc="white")
ax.set_xscale("log")
ax.set_xticks([40, 60, 156, 300])
ax.set_xticks([], minor=True)
ax.set_xticklabels(["40\nM5", "60\nM1", "156\nM5-full", "300\nM5-300"], fontsize=6.5)
ax.set_xlabel("training hours")
ax.set_ylabel("WER (%)")
ax.grid(lw=0.3, alpha=0.5)
ax.set_ylim(9, 20.5)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))  # hollow square = M1 (caption)
fig.tight_layout()
save(fig, "f16_data_scale", d)
