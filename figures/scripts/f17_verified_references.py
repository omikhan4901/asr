"""
F17: WER against the original vs the human-verified reference on the verified clips (305 DL Sprint eval, 100 each
FLEURS, Common Voice, Kathbath), for the final model and the main comparison systems. Hollow = original reference,
filled = verified; the number is the paired change (verified minus original) with its 95% cluster-bootstrap CI.

Inputs: results/labels/summary.json (analysis/labeling/import_labels.py)
Run:    PYTHONPATH=src .venv/bin/python figures/scripts/f17_verified_references.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"dlsprint": "DL Sprint eval", "fleurs": "FLEURS", "commonvoice": "Common Voice", "kathbath": "Kathbath valid"}
SYSTEMS = [  # label, key in summary.json, colour
    ("tugstugi zero-shot", "modal/zeroshot/tugstugi_a100", OKABE["grey"]),
    ("Hishab conformer large", "zeroshot/hishab__titu_stt_bn_conformer_large", OKABE["grey"]),
    ("IndicConformer CTC", "kaggle/zeroshot/ai4bharat__indic-conformer-600m-multilingual_ctc_t4", OKABE["grey"]),
    ("Team Villagers (DL Sprint 4.0)", "kaggle/zeroshot/teamvillagers__whisper-bengali-ct2_t4", OKABE["grey"]),
    ("M1 (DL Sprint, 60 h)", "modal/finetuned/m1_full", OKABE["sky"]),
    ("Final model (ours)", "kaggle/finetuned/soup_m1_full+m5_300+m5_300+m5_300_t4", OKABE["blue"]),
]


s = json.loads((ROOT / "results/labels/summary.json").read_text(encoding="utf-8"))["refs"]["by_source"]
rows = []
for src, name in SETS.items():
    models = s[src]["models"]
    for label, key, _ in SYSTEMS:
        m = models[key]  # key = results root (relative to results/) + system folder
        rows.append({"set": name, "system": label, "utts": m["utts"], "wer_original": 100 * m["wer_original"],
                     "wer_verified": 100 * m["wer_verified"], "diff": 100 * m["diff_verified_minus_original"],
                     "diff_lo": 100 * m["diff_ci95"][0], "diff_hi": 100 * m["diff_ci95"][1]})
d = pd.DataFrame(rows)

fig, axes = plt.subplots(1, len(SETS), figsize=(PAGE_W, 2.3), sharey=True)
for ax, name in zip(axes, SETS.values()):
    g = d[d.set == name].set_index("system")
    for yi, (label, _, col) in enumerate(SYSTEMS):
        r = g.loc[label]
        ax.plot([r.wer_original, r.wer_verified], [yi, yi], color=col, lw=1, zorder=1)
        ax.scatter(r.wer_original, yi, s=16, facecolor="white", edgecolor=col, lw=0.9, zorder=2)
        ax.scatter(r.wer_verified, yi, s=16, color=col, zorder=3)
        sig = not (r.diff_lo <= 0 <= r.diff_hi)
        ax.text(36, yi, f"{r['diff']:+.1f}", va="center", ha="right", fontsize=6.5,
                color="0.15" if sig else "0.55", fontweight="bold" if sig else "normal")
    ax.set_title(f"{name} (n={int(g['utts'].iloc[0])})")
    ax.set_xlim(0, 36.5)
    ax.set_xlabel("WER (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(range(len(SYSTEMS)))
axes[0].set_yticklabels([s_[0] for s_ in SYSTEMS])
fig.text(0.99, -0.1, "hollow = original reference, filled = human-verified; right: change in points "
         "(bold = 95% CI excludes 0)", ha="right", fontsize=6.5)
save(fig, "f17_verified_references", d)
