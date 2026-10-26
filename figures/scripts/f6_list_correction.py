"""
F6: raising the ceiling without retraining (list correction). Per test set: (a) WER of greedy decoding and of
LM-weighted MBR on the expanded candidate list (95% CI of the paired difference, carried over to the
WER); (b) oracle WER of the candidate pool before and after adding the learned spelling rewrites.
All settings tuned on DL Sprint dev and applied unchanged.

Input: results/list_correction/summary.csv
Run:   ./Scripts/python.exe figures/scripts/f6_list_correction.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-dev": "DL Sprint dev (tuning)", "splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS",
        "kathbath-valid": "Kathbath valid"}
d = pd.read_csv(ROOT / "results/list_correction/summary.csv")
d = d[d.set.isin(SETS)].set_index("set").reindex(SETS)
out = pd.DataFrame({"set": list(SETS.values()), "greedy": 100 * d.greedy_wer.values,
                    "list_correction": 100 * d.mbr_lm_x_wer.values,
                    "list_correction_lo": 100 * (d.greedy_wer + d.mbr_lm_x_ci_lo).values,
                    "list_correction_hi": 100 * (d.greedy_wer + d.mbr_lm_x_ci_hi).values,
                    "oracle_pool": 100 * d.oracle_pool_wer.values, "oracle_expanded": 100 * d.oracle_x_wer.values})

fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.0), sharey=True)
y = list(range(len(out)))[::-1]
panels = [("greedy", "list_correction", "(a) WER: greedy → LM-MBR on expanded list", "greedy", "list correction"),
          ("oracle_pool", "oracle_expanded", "(b) oracle WER: pool → expanded pool", "pool", "expanded")]
for ax, (a, b, title, la, lb) in zip(axes, panels):
    for yi, (_, r) in zip(y, out.iterrows()):
        ax.plot([r[a], r[b]], [yi, yi], color=OKABE["grey"], lw=1, zorder=1)
    ax.scatter(out[a], y, color=OKABE["grey"], s=18, zorder=2, label=la)
    ax.scatter(out[b], y, color=OKABE["blue"], s=18, zorder=3, label=lb)
    if b == "list_correction":
        ax.errorbar(out[b], y, xerr=[out[b] - out.list_correction_lo, out.list_correction_hi - out[b]], fmt="none",
                    ecolor=OKABE["blue"], lw=0.8, capsize=1.5, zorder=3)
    for yi, (_, r) in zip(y, out.iterrows()):
        ax.text(r[b] - 0.4, yi + 0.18, f"{r[b]:.1f}", ha="right", fontsize=6.5, color=OKABE["blue"])
        ax.text(r[a] + 0.4, yi + 0.18, f"{r[a]:.1f}", ha="left", fontsize=6.5, color=OKABE["grey"])
    ax.set_title(title, loc="left")
    ax.set_xlabel("WER (%)")
    ax.set_xlim(0, 33)
    ax.legend(frameon=False, loc="lower right")
axes[0].set_yticks(y)
axes[0].set_yticklabels(out.set)
save(fig, "f6_list_correction", out)
