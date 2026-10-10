"""
F8: ceilings on DL Sprint eval (tugstugi). (a) What selection can reach and what each intervention
reached: greedy, MBR, best single candidate (utterance oracle) and the word-coverage bound of the
candidate pool (share of reference words no candidate contains: no selection or word-level combination
can go below it), before and after adding spelling rewrites; then the Phase 3 and fine-tuned systems.
(b) What the selection-proof errors of greedy (no candidate has the reference word) are, by rule-based type.

Inputs: results/phase3/summary.csv, results/agreement/eval_sampled_tugstugi/summary.json,
        results/error_types/with_nbest/eval_sampled/summary.json, results/modal/finetuned/splits_fa-eval/m1_full/metrics.json
Run:    ./Scripts/python.exe figures/scripts/f8_ceilings.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

p3 = pd.read_csv(ROOT / "results/phase3/summary.csv").set_index("set").loc["splits_fa-eval"]
ag = json.loads((ROOT / "results/agreement/eval_sampled_tugstugi/summary.json").read_text(encoding="utf-8"))
et = json.loads((ROOT / "results/error_types/with_nbest/eval_sampled/summary.json").read_text(encoding="utf-8"))
ft = json.loads((ROOT / "results/modal/finetuned/splits_fa-eval/m1_full/metrics.json").read_text(encoding="utf-8"))

bars = [("greedy decoding", 100 * p3.greedy_wer, "system"),
        ("MBR (12 sampled candidates)", 100 * ag["wer_mbr"], "system"),
        ("best single candidate (oracle)", 100 * p3.oracle_pool_wer, "bound"),
        ("words in no candidate (coverage bound)", 100 * p3.coverage_miss_pool, "bound"),
        ("oracle after adding spelling rewrites", 100 * p3.oracle_x_wer, "bound expanded"),
        ("coverage bound after rewrites", 100 * p3.coverage_miss_x, "bound expanded"),
        ("Phase 3: LM-MBR on expanded list", 100 * p3.mbr_lm_x_wer, "achieved"),
        ("Phase 4: M1 full fine-tune", 100 * ft["wer"], "achieved")]
COL = {"system": OKABE["grey"], "bound": OKABE["sky"], "bound expanded": OKABE["green"], "achieved": OKABE["blue"]}
types = et["selection_proof"]["type_share_of_selection_proof"]
NAMES = {"convention": "convention", "vowel_shift": "vowel shift", "segmentation": "segmentation",
         "near_miss": "near miss", "substitution": "substitution", "deletion": "deletion", "number": "number"}
TCOL = {"convention": OKABE["orange"], "vowel_shift": OKABE["yellow"], "segmentation": OKABE["red"],
        "near_miss": OKABE["sky"], "substitution": OKABE["blue"], "deletion": OKABE["grey"], "number": OKABE["purple"]}

fig, (ax, bx) = plt.subplots(2, 1, figsize=(PAGE_W, 3.6), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.75})
y = list(range(len(bars)))[::-1]
for yi, (label, v, kind) in zip(y, bars):
    ax.barh(yi, v, color=COL[kind])
    ax.text(v + 0.2, yi, f"{v:.2f}", va="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars])
ax.set_xlabel("WER (%) on DL Sprint eval (coverage bounds: % of reference words)")
ax.set_xlim(0, 26)
ax.set_title("(a) ceilings of selection, and what each intervention reached", loc="left")

left = 0.0
for k, name in NAMES.items():
    share = 100 * types.get(k, 0.0)
    bx.barh(0, share, left=left, color=TCOL[k], edgecolor="white", lw=0.5)
    if share >= 4:
        bx.text(left + share / 2, 0, f"{name.replace(' ', chr(10))}\n{share:.0f}%", ha="center", va="center",
                fontsize=6)
    left += share
bx.set_xlim(0, 100)
bx.set_yticks([])
bx.set_xlabel("% of greedy's selection-proof errors (variant types: convention, vowel shift, segmentation)")
bx.set_title("(b) what the errors no candidate can fix are (rule-based types)", loc="left")
save(fig, "f8_ceilings", pd.DataFrame([{"item": b[0], "value": b[1], "kind": b[2]} for b in bars]
                                       + [{"item": f"selection-proof {k}", "value": 100 * v, "kind": "type"}
                                          for k, v in types.items()]))
