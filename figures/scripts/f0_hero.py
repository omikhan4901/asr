"""
F0 (overview): the paper in one figure. (a) DL Sprint eval WER from zero-shot greedy through MBR rescoring, the
selection ceiling (best of 12 candidates), list correction (no retraining) to the final fine-tuned model. (b) Each
test set, zero-shot vs the final model, with 95% cluster-bootstrap CIs.

Inputs: figures/data/f8_ceilings.csv (run f8_ceilings.py first), results/modal/zeroshot/<set>/tugstugi_a100/metrics.json,
        results/kaggle/finetuned/<set>/<FINAL>/metrics.json
Run:    ./Scripts/python.exe figures/scripts/f0_hero.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

FINAL = "soup_m1_full+m5_300+m5_300+m5_300_t4"  # the final model: 0.25 M1 + 0.75 M5-300, chosen by the dev rule (PROGRESS §70)
SETS = [("splits_fa-eval", "DL Sprint eval"), ("fleurs-test", "FLEURS"), ("cv-test", "Common Voice"),
        ("kathbath-valid", "Kathbath valid")]


def metrics(p):
    m = json.loads((ROOT / p / "metrics.json").read_text(encoding="utf-8"))
    return 100 * m["wer"], 100 * m["wer_ci95"][0], 100 * m["wer_ci95"][1]


c = pd.read_csv(ROOT / "figures/data/f8_ceilings.csv").set_index("item")["value"]
final_eval = metrics(f"results/kaggle/finetuned/splits_fa-eval/{FINAL}")[0]
steps = [("zero-shot\ngreedy", c["greedy decoding"], OKABE["grey"]),
         ("MBR\nrescoring", c["MBR (12 sampled candidates)"], OKABE["grey"]),
         ("best of 12\n(ceiling)", c["best single candidate (oracle)"], None),
         ("list\ncorrection\n(no retraining)", c["List correction: LM-MBR on expanded list"], OKABE["orange"]),
         ("final\nmodel\n(fine-tuned)", final_eval, OKABE["blue"])]
rows = [{"panel": "a", "item": lab.replace("\n", " "), "wer": round(v, 3), "lo": None, "hi": None} for lab, v, _ in steps]

fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
for i, (lab, v, col) in enumerate(steps):
    if col is None:
        a.bar(i, v, 0.7, fill=False, ls="--", lw=0.8, edgecolor=OKABE["grey"])
    else:
        a.bar(i, v, 0.7, color=col)
    a.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
a.set_xticks(range(len(steps)))
a.set_xticklabels([s[0] for s in steps], fontsize=6.5)
a.set_ylabel("WER (%)")
a.set_ylim(0, 26)
a.set_title("(a) DL Sprint eval", loc="left")
a.grid(axis="y", lw=0.3, alpha=0.5)

for yi, (s, name) in enumerate(SETS[::-1]):
    z, zl, zh = metrics(f"results/modal/zeroshot/{s}/tugstugi_a100")
    f, fl, fh = metrics(f"results/kaggle/finetuned/{s}/{FINAL}")
    b.plot([f, z], [yi, yi], color="0.8", lw=1.5, zorder=1)
    b.errorbar(z, yi, xerr=[[z - zl], [zh - z]], fmt="o", ms=4, lw=0.8, capsize=1.5, color=OKABE["grey"],
               label="zero-shot" if yi == 0 else None)
    b.errorbar(f, yi, xerr=[[f - fl], [fh - f]], fmt="o", ms=4, lw=0.8, capsize=1.5, color=OKABE["blue"],
               label="final model" if yi == 0 else None)
    b.text(z + 0.8, yi + 0.18, f"{z:.1f}", fontsize=6.5, color="0.35")
    b.text(f - 0.8, yi + 0.18, f"{f:.1f}", fontsize=6.5, color=OKABE["blue"], ha="right")
    rows += [{"panel": "b", "item": f"{name}: zero-shot", "wer": round(z, 3), "lo": round(zl, 3), "hi": round(zh, 3)},
             {"panel": "b", "item": f"{name}: final", "wer": round(f, 3), "lo": round(fl, 3), "hi": round(fh, 3)}]
b.set_yticks(range(len(SETS)))
b.set_yticklabels([n for _, n in SETS[::-1]])
b.set_xlim(5, 25)
b.set_ylim(-0.5, len(SETS) - 0.3)
b.set_xlabel("WER (%)")
b.set_title("(b) four test sets", loc="left")
b.grid(axis="x", lw=0.3, alpha=0.5)
b.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False)
fig.tight_layout()
save(fig, "f0_hero", pd.DataFrame(rows))
