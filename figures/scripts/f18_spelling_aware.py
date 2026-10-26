"""
F18: standard WER (hollow) vs spelling-aware WER (filled) on the four full test sets: edits the human labels confirmed
as spelling only (convention, segmentation) are not counted. Error bars = 95% cluster-bootstrap CI of the
spelling-aware WER. Rows trained on the test material are named in the caption (not compared).

Inputs: results/spelling_aware/summary.csv (analysis/spelling_aware_wer.py)
Run:    PYTHONPATH=src .venv/bin/python figures/scripts/f18_spelling_aware.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "cv-test": "Common Voice",
        "kathbath-valid": "Kathbath valid"}
SYSTEMS = [("tugstugi zero-shot", OKABE["grey"]), ("zarifmahir21", OKABE["grey"]), ("Team Villagers", OKABE["grey"]),
           ("IndicConformer CTC", OKABE["grey"]), ("Hishab conformer large", OKABE["grey"]),
           ("M1 (DL Sprint, 60 h)", OKABE["sky"]), ("Final model", OKABE["blue"])]
CONTAMINATED = {("splits_fa-eval", "zarifmahir21"), ("splits_fa-eval", "Team Villagers"),
                ("cv-test", "Hishab conformer large"), ("kathbath-valid", "Hishab conformer large")}

S = pd.read_csv(ROOT / "results/spelling_aware/summary.csv")
d = S[S.system.isin([s for s, _ in SYSTEMS])].copy()
for c in ("wer", "sa_wer", "sa_wer_lo", "sa_wer_hi"):
    d[c] = 100 * d[c]
d["contaminated"] = [(s, m) in CONTAMINATED for s, m in zip(d.set, d.system)]

fig, axes = plt.subplots(1, len(SETS), figsize=(PAGE_W, 2.4), sharey=True)
for ax, (s, name) in zip(axes, SETS.items()):
    g = d[d.set == s].set_index("system")
    for yi, (label, col) in enumerate(SYSTEMS):
        r = g.loc[label]
        ax.plot([r.wer, r.sa_wer], [yi, yi], color=col, lw=1, zorder=1)
        ax.scatter(r.wer, yi, s=16, facecolor="white", edgecolor=col, lw=0.9, zorder=2)
        ax.errorbar(r.sa_wer, yi, xerr=[[r.sa_wer - r.sa_wer_lo], [r.sa_wer_hi - r.sa_wer]], fmt="o", ms=4, lw=0.8,
                    capsize=1.5, color=col, zorder=3)
    ax.set_title(name)
    ax.set_xlim(0, 32)
    ax.set_xlabel("WER (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(range(len(SYSTEMS)))
axes[0].set_yticklabels([s for s, _ in SYSTEMS])
fig.text(0.99, -0.09, "hollow = standard WER, filled = spelling-aware WER with 95% CI. Not comparable (trained on that material): "
         "zarifmahir21 and Team Villagers on DL Sprint, Hishab on CV and Kathbath", ha="right", fontsize=6.5)
save(fig, "f18_spelling_aware", d)
