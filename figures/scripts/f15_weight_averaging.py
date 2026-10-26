"""
F15: model selection by weight averaging. (a) Each pre-registered candidate's average WER over the three dev sets
against its average over the four test sets; the rule picks the lowest dev average. (b) Per-set paired difference of
the chosen average (0.25 M1 + 0.75 M5-300) against M5-300 alone, with 95% cluster-bootstrap CIs.

Inputs: results/kaggle/finetuned/<set>/<candidate>/metrics.json, results/compare/soup31_vs_m5_300_<set>.json
Run:    ./Scripts/python.exe figures/scripts/f15_weight_averaging.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

DEV = {"splits_fa-dev": "DL Sprint dev", "cv-dev": "CV dev", "fleurs-validation": "FLEURS val"}
TEST = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "cv-test": "Common Voice",
        "kathbath-valid": "Kathbath valid"}
CANDS = {"m5_300_t4": "M5-300 alone", "soup_m5_full+m5_300_t4": "M5-full + M5-300",
         "soup_m1_full+m5_300_t4": "M1 + M5-300", "soup_m1_full+m5_300+m5_300+m5_300_t4": "0.25 M1 + 0.75 M5-300"}
CHOSEN = "soup_m1_full+m5_300+m5_300+m5_300_t4"


def wer(s, c):
    return 100 * json.loads((ROOT / f"results/kaggle/finetuned/{s}/{c}/metrics.json").read_text(encoding="utf-8"))["wer"]


rows = [{"panel": "a", "candidate": name, "dev_avg": sum(wer(s, c) for s in DEV) / len(DEV),
         "test_avg": sum(wer(s, c) for s in TEST) / len(TEST)} for c, name in CANDS.items()]
for s, name in {**DEV, **TEST}.items():
    r = json.loads((ROOT / f"results/compare/soup31_vs_m5_300_{s}.json").read_text(encoding="utf-8"))
    rows.append({"panel": "b", "set": name, "diff": 100 * r["diff_b_minus_a"], "lo": 100 * r["diff_ci95"][0],
                 "hi": 100 * r["diff_ci95"][1], "split": "dev" if s in DEV else "test"})
d = pd.DataFrame(rows)

fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE_W, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})
pa = d[d.panel == "a"]
for c, (_, r) in zip(CANDS, pa.iterrows()):
    col = OKABE["blue"] if c == CHOSEN else OKABE["grey"]
    a.scatter(r.dev_avg, r.test_avg, s=22, color=col, zorder=3)
    a.annotate(r.candidate, (r.dev_avg, r.test_avg), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=col)
a.set_xlabel("dev average WER (%), used to choose")
a.set_ylabel("test average WER (%)")
a.set_title("(a) choice on dev vs result on test", loc="left")
a.grid(lw=0.3, alpha=0.5)
a.set_xlim(pa.dev_avg.min() - 0.1, pa.dev_avg.max() + 0.35)

pb = d[d.panel == "b"].reset_index(drop=True)
for yi, r in pb[::-1].reset_index(drop=True).iterrows():
    col = OKABE["orange"] if r.split == "dev" else OKABE["blue"]
    b.errorbar(r["diff"], yi, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", ms=3.5, lw=0.8, capsize=1.5, color=col)
b.axvline(0, color="0.4", lw=0.6)
b.set_yticks(range(len(pb)))
b.set_yticklabels(pb.set[::-1])
b.set_xlabel("WER difference vs M5-300 (points; < 0 = better)")
b.set_title("(b) chosen average vs M5-300 (orange = dev)", loc="left")
b.grid(axis="x", lw=0.3, alpha=0.5)
fig.tight_layout()
save(fig, "f15_weight_averaging", d)
