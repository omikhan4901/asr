"""
F3: share of each model's word errors that are spelling/dialect variants
(rule-based typer), against its WER, per test set, with 95% cluster-bootstrap CIs.
Rows a model trained on are left out (contamination, PROGRESS 27 and 33).

Input: results/error_types/summary.csv
Run:   ./Scripts/python.exe figures/scripts/f3_variant_share_by_model.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import COL_W, ROOT, SET_COLOR, plt, save  # noqa: E402

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "kathbath-valid": "Kathbath valid"}
CONTAMINATED = {("kathbath-valid", "hishab__titu_stt_bn_conformer_large"),
                ("kathbath-valid", "hishab__titu_stt_bn_fastconformer")}
d = pd.read_csv(ROOT / "results/error_types/summary.csv", encoding="utf-8-sig")
d = d[d.set.isin(SETS) & ~d.apply(lambda r: (r.set, r.model) in CONTAMINATED, axis=1)].copy()
ci = d["variant_share_ci"].str.strip("[]").str.split(",", expand=True).astype(float)
d["lo"], d["hi"], d["set_name"] = ci[0], ci[1], d.set.map(SETS)

fig, ax = plt.subplots(figsize=(COL_W, 2.2))
for name, g in d.groupby("set_name"):
    ax.errorbar(g["WER%"], g["variant_share%"], yerr=[g["variant_share%"] - g.lo, g.hi - g["variant_share%"]],
                fmt="o", ms=3, lw=0.8, capsize=1.5, color=SET_COLOR[name], label=name)
ax.set_xscale("log")
ax.set_xticks([10, 20, 40, 80])
ax.set_xticklabels(["10", "20", "40", "80"])
ax.set_xlabel("WER (%), log scale")
ax.set_ylabel("variant share of errors (%)")
ax.set_ylim(0, 60)
ax.legend(frameon=False, loc="upper right")
save(fig, "f3_variant_share_by_model",
     d[["set", "model", "WER%", "variant_share%", "lo", "hi"]].rename(columns={"WER%": "wer", "variant_share%": "variant_share"}))
