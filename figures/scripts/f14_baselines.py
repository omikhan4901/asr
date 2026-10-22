"""
F14: the final model against the open Bangla models scored with the same harness (greedy), one panel per test set,
WER with 95% cluster-bootstrap CIs. Hollow marker = the model trained on that material (not compared). IndicConformer
is shown with both heads (its RNN-T decoder returns nothing on many clips over 15 s). psdn-ai/bangla-asr is left out
(63-76% WER: it writes case endings as separate words).

Inputs: results/kaggle/finetuned/<set>/<FINAL>/metrics.json, results/kaggle/zeroshot/<set>/<baseline>_t4/metrics.json,
        results/zeroshot/<set>/hishab__titu_stt_bn_conformer_large/, results/modal/zeroshot/<set>/tugstugi_a100/
Run:    ./Scripts/python.exe figures/scripts/f14_baselines.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

FINAL = "soup_m1_full+m5_300+m5_300+m5_300_t4"
SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "cv-test": "Common Voice",
        "kathbath-valid": "Kathbath valid"}
K = "results/kaggle/zeroshot/{s}/"
SYSTEMS = [  # label, folder, colour
    ("tugstugi zero-shot", "results/modal/zeroshot/{s}/tugstugi_a100", OKABE["grey"]),
    ("zarifmahir21", K + "zarifmahir21__whisper-medium-bangla_t4", OKABE["grey"]),
    ("Team Villagers (DL Sprint 4.0)", K + "teamvillagers__whisper-bengali-ct2_t4", OKABE["grey"]),
    ("IndicConformer RNN-T", K + "ai4bharat__indic-conformer-600m-multilingual_t4", OKABE["grey"]),
    ("IndicConformer CTC", K + "ai4bharat__indic-conformer-600m-multilingual_ctc_t4", OKABE["grey"]),
    ("Hishab conformer large", "results/zeroshot/{s}/hishab__titu_stt_bn_conformer_large", OKABE["grey"]),
    ("Final model (ours)", "results/kaggle/finetuned/{s}/" + FINAL, OKABE["blue"]),
]
# trained on that material (declared before scoring, PROGRESS §68)
CONTAMINATED = {("splits_fa-eval", "zarifmahir21"), ("splits_fa-eval", "Team Villagers (DL Sprint 4.0)"),
                ("cv-test", "Hishab conformer large"), ("kathbath-valid", "Hishab conformer large")}

rows = []
for s, sname in SETS.items():
    for label, folder, _ in SYSTEMS:
        m = json.loads((ROOT / folder.format(s=s) / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"set": sname, "system": label, "wer": 100 * m["wer"], "lo": 100 * m["wer_ci95"][0],
                     "hi": 100 * m["wer_ci95"][1], "contaminated": (s, label) in CONTAMINATED})
d = pd.DataFrame(rows)

fig, axes = plt.subplots(1, len(SETS), figsize=(PAGE_W, 2.4), sharey=True)
for ax, sname in zip(axes, SETS.values()):
    g = d[d.set == sname].set_index("system")
    for yi, (label, _, col) in enumerate(SYSTEMS):
        r = g.loc[label]
        ax.errorbar(r.wer, yi, xerr=[[r.wer - r.lo], [r.hi - r.wer]], fmt="o", ms=3.5, lw=0.8, capsize=1.5,
                    color=col, mfc="white" if r.contaminated else col)
    ax.axvline(g.loc["Final model (ours)"].wer, color=OKABE["blue"], lw=0.5, ls=":", zorder=0)
    ax.set_title(sname)
    ax.set_xlim(0, 35)
    ax.set_xlabel("WER (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(range(len(SYSTEMS)))
axes[0].set_yticklabels([s[0] for s in SYSTEMS])
fig.text(0.99, -0.09, "hollow marker = the model was trained on this material (not compared); "
         "psdn-ai/bangla-asr not shown (63-76% WER, splits case endings)", ha="right", fontsize=6.5)
save(fig, "f14_baselines", d)
