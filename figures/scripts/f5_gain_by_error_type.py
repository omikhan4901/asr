"""
F5: DL Sprint eval WER split into spelling/dialect-variant errors and all other errors (rule-based
typer), for zero-shot tugstugi, the Phase 3 system without training (LM-MBR on the expanded
candidate list) and the four M1 fine-tuning variants.

Inputs: results/error_types/phase4/splits_fa-eval/<run>/summary.json (A100 rows),
        results/error_types/phase3/splits_fa-eval/mbr_lm_x/summary.json (Phase 3, RTX 3060)
Run:    ./Scripts/python.exe figures/scripts/f5_gain_by_error_type.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import COL_W, OKABE, ROOT, plt, save  # noqa: E402

SYSTEMS = [("zero-shot", "results/error_types/phase4/splits_fa-eval/tugstugi_a100"),
           ("List correction (no retraining)", "results/error_types/phase3/splits_fa-eval/mbr_lm_x"),
           ("encoder-only", "results/error_types/phase4/splits_fa-eval/m1_encoder"),
           ("decoder-only", "results/error_types/phase4/splits_fa-eval/m1_decoder"),
           ("LoRA", "results/error_types/phase4/splits_fa-eval/m1_lora"),
           ("full", "results/error_types/phase4/splits_fa-eval/m1_full")]

rows = []
for label, folder in SYSTEMS:
    m = json.loads((ROOT / folder / "summary.json").read_text(encoding="utf-8"))
    wer = 100 * m["wer"]["value"]
    v = m["variant_share"]["value"]
    rows.append({"system": label, "wer": wer, "variant_points": wer * v, "other_points": wer * (1 - v)})
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(COL_W, 2.3))
y = range(len(d))[::-1]
ax.barh(list(y), d.variant_points, color=OKABE["orange"], label="spelling / dialect variants")
ax.barh(list(y), d.other_points, left=d.variant_points, color=OKABE["blue"], label="other errors")
for yi, (_, r) in zip(y, d.iterrows()):
    ax.text(r.wer + 0.3, yi, f"{r.wer:.1f}", va="center", fontsize=7)
ax.set_yticks(list(y))
ax.set_yticklabels(d.system)
ax.set_xlabel("WER (%) on DL Sprint eval")
ax.set_xlim(0, 26)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0))
save(fig, "f5_gain_by_error_type", d)
