"""
F13: train x test matrix. Rows = training data (zero-shot tugstugi, M1 DL Sprint, M2 CV, M3 Kathbath,
M4 Shrutilipi, M5 merged 10 h x 4; same recipe: full FT, lr 1e-5, 2 epochs), columns = test sets + the
macro-average of the four. Cell text = WER and the paired difference vs zero-shot [95% cluster CI];
shade = relative WER reduction vs zero-shot in that column (columns differ in difficulty). The best
model per column is outlined. M1 scored on the Modal A100, M2-M5 on Kaggle T4 (T4 = A100 within 0.01 WER
points, results/compare/gpu_t4/).

Input: results/compare/finetuning/<set>__<run>_vs_tugstugi_a100.json (runs/finetuning_analysis.sh)
Run:   ./Scripts/python.exe figures/scripts/f13_matrix_heatmap.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "cv-test": "Common Voice",
        "kathbath-valid": "Kathbath valid"}
RUNS = {"zero-shot": "zero-shot (no training)", "m1_full": "M1 DL Sprint (59.6 h)",
        "m2_cv_t4": "M2 Common Voice (34.4 h)", "m3_kathbath_t4": "M3 Kathbath (33.9 h)",
        "m4_shrutilipi_t4": "M4 Shrutilipi (28.1 h)", "m5_merged_t4": "M5 merged (10 h x 4)",
        "m6_harmonized_t4": "M6 merged, harmonized"}
# a model without all four comparisons yet (e.g. M6 before its Kaggle scoring) is left out
RUNS = {r: n for r, n in RUNS.items() if r == "zero-shot" or all(
    (ROOT / "results/compare/finetuning" / f"{s}__{r}_vs_tugstugi_a100.json").exists() for s in SETS)}

rows = []
for s, sname in SETS.items():
    for run, rname in RUNS.items():
        if run == "zero-shot":
            continue
        d = json.loads((ROOT / "results/compare/finetuning" / f"{s}__{run}_vs_tugstugi_a100.json").read_text(encoding="utf-8"))
        rows.append({"train": rname, "test": sname, "wer": 100 * d["wer_b"], "diff": 100 * d["diff_b_minus_a"],
                     "lo": 100 * d["diff_ci95"][0], "hi": 100 * d["diff_ci95"][1], "zero_shot": 100 * d["wer_a"],
                     "n_utt": d["n_utterances"]})
    rows.append({"train": RUNS["zero-shot"], "test": sname, "wer": rows[-1]["zero_shot"], "diff": 0.0, "lo": 0.0,
                 "hi": 0.0, "zero_shot": rows[-1]["zero_shot"], "n_utt": rows[-1]["n_utt"]})
d = pd.DataFrame(rows)
macro = d.groupby("train", sort=False).agg(wer=("wer", "mean"), zero_shot=("zero_shot", "mean")).reset_index()
macro["test"], macro["diff"] = "Macro-average", macro.wer - macro.zero_shot
d = pd.concat([d, macro], ignore_index=True)
d["rel_reduction"] = 100 * (d.zero_shot - d.wer) / d.zero_shot

cols, rlabels = list(SETS.values()) + ["Macro-average"], list(RUNS.values())
W = d.pivot(index="train", columns="test", values="wer").loc[rlabels, cols]
R = d.pivot(index="train", columns="test", values="rel_reduction").loc[rlabels, cols]
key = d.set_index(["train", "test"])

fig, ax = plt.subplots(figsize=(PAGE_W, 0.46 * len(RUNS)))
im = ax.imshow(R.values, cmap="Blues", vmin=0, vmax=55, aspect="auto")
for i, r in enumerate(rlabels):
    for j, c in enumerate(cols):
        row = key.loc[(r, c)]
        dark = R.values[i, j] > 32
        best = W[c].idxmin() == r
        ax.text(j, i - 0.13, f"{row.wer:.2f}", ha="center", va="center", fontsize=8,
                fontweight="bold" if best else "normal", color="white" if dark else "black")
        if r != rlabels[0]:
            sub = f"{row['diff']:+.2f}" if c == "Macro-average" else f"{row['diff']:+.2f} [{row.lo:+.1f}, {row.hi:+.1f}]"
            ax.text(j, i + 0.22, sub.replace("-", "−"), ha="center", va="center", fontsize=5.6, color="white" if dark else "#333333")
        if best:
            ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, lw=1.3, ec="black", zorder=5))
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(range(len(rlabels)))
ax.set_yticklabels(rlabels)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.axvline(len(SETS) - 0.5, color="white", lw=3)
ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
ax.set_yticks(np.arange(-0.5, len(rlabels)), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("WER reduction vs zero-shot (%)", fontsize=7)
cb.outline.set_visible(False)
save(fig, "f13_matrix_heatmap", d.round(4))
