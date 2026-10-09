"""
F2: 1-best errors by agreement count c, and the share MBR fixes (with 95%
cluster-bootstrap CIs over recordings). DL Sprint eval, tugstugi sampled 12-best,
MBR with exact scores (registry R23).

Input: results/agreement/eval_sampled_tugstugi/errors.csv
Run:   ./Scripts/python.exe figures/scripts/f2_mbr_fix_by_agreement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import COL_W, OKABE, ROOT, plt, save  # noqa: E402

err = pd.read_csv(ROOT / "results/agreement/eval_sampled_tugstugi/errors.csv", encoding="utf-8-sig")
err["fixed"] = err["mbr_correct"].astype(bool)
groups = err["group"].unique()
rng = np.random.default_rng(0)
by_g = {c: err[err.c == c].groupby("group")["fixed"].agg(["sum", "count"]).reindex(groups, fill_value=0)
        for c in sorted(err.c.unique())}
rows = []
for c, t in by_g.items():
    s, n = t["sum"].to_numpy(float), t["count"].to_numpy(float)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    rates = s[draws].sum(1) / np.maximum(1, n[draws].sum(1))
    rows.append({"c": c, "errors": int(n.sum()), "fixed": int(s.sum()), "fix_rate": s.sum() / n.sum(),
                 "fix_lo": np.percentile(rates, 2.5), "fix_hi": np.percentile(rates, 97.5)})
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(COL_W, 2.1))
ax.bar(d.c, d.errors - d.fixed, color="#C9C9C9", width=0.75, label="not fixed by MBR")
ax.bar(d.c, d.fixed, bottom=d.errors - d.fixed, color=OKABE["blue"], width=0.75, label="fixed by MBR")
ax.set_xlabel("c = candidates (of 12) with the reference word right")
ax.set_ylabel("1-best word errors")
ax.set_xticks(d.c)
ax.set_ylim(0, d.errors.max() * 1.12)
ax2 = ax.twinx()
ax2.errorbar(d.c, 100 * d.fix_rate, yerr=[100 * (d.fix_rate - d.fix_lo), 100 * (d.fix_hi - d.fix_rate)],
             color=OKABE["red"], marker="o", ms=2.5, lw=0.9, capsize=1.5, label="share fixed (%)")
ax2.set_ylim(0, 105)
ax2.set_ylabel("share fixed by MBR (%)", color=OKABE["red"])
ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0))
save(fig, "f2_mbr_fix_by_agreement", d)
