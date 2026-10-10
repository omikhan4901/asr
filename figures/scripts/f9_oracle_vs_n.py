"""
F9: does a longer candidate list help? Oracle WER (best of the first N candidates, averaged over 20
random orderings of the 12 sampled candidates) against N, for tugstugi on four sets, with 95%
cluster-bootstrap CIs at each N; dashed line = greedy decoding on the same set.

Inputs: data/nbest_fa/{dev,eval}_sampled.jsonl, data/nbest_ext/{fleurs,kathbath}_tugstugi.jsonl,
        results/zeroshot/<set>/<tugstugi>/predictions.jsonl
Run:    PYTHONPATH=src ./Scripts/python.exe figures/scripts/f9_oracle_vs_n.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from style import COL_W, OKABE, ROOT, plt, save  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio, edit_counts  # noqa: E402

TUG = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
SETS = {"DL Sprint eval": ("data/nbest_fa/eval_sampled.jsonl", "splits_fa-eval", OKABE["blue"]),
        "DL Sprint dev": ("data/nbest_fa/dev_sampled.jsonl", "splits_fa-dev", OKABE["sky"]),
        "FLEURS": ("data/nbest_ext/fleurs_tugstugi.jsonl", "fleurs-test", OKABE["orange"]),
        "Kathbath valid": ("data/nbest_ext/kathbath_tugstugi.jsonl", "kathbath-valid", OKABE["green"])}
N_ORDERS, K = 20, 12
rng = np.random.default_rng(0)

rows = []
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
for name, (nb, s, color) in SETS.items():
    recs = [json.loads(l) for l in open(ROOT / nb, encoding="utf-8")]
    edits = np.full((len(recs), K), np.nan)
    refw, groups = np.zeros(len(recs)), []
    for i, r in enumerate(recs):
        for k, c in enumerate(r["candidates"][:K]):
            e = edit_counts(c["text"], r["ground_truth"])
            edits[i, k], refw[i] = e["word_edits"], e["ref_words"]
        groups.append(r["group"])
    edits = np.where(np.isnan(edits), np.nanmax(edits, axis=1, keepdims=True), edits)
    exp_min = np.zeros((len(recs), K))
    for _ in range(N_ORDERS):
        perm = edits[:, rng.permutation(K)]
        exp_min += np.minimum.accumulate(perm, axis=1)
    exp_min /= N_ORDERS
    for n in range(1, K + 1):
        b = bootstrap_ratio(groups, exp_min[:, n - 1], refw, n_boot=1000)
        rows.append({"set": name, "n": n, "wer": 100 * b["value"], "lo": 100 * b["ci95"][0], "hi": 100 * b["ci95"][1]})
    g = pd.DataFrame([r for r in rows if r["set"] == name])
    ax.plot(g.n, g.wer, "-o", ms=2.5, lw=1, color=color, label=name)
    ax.fill_between(g.n, g.lo, g.hi, color=color, alpha=0.15, lw=0)
    greedy = [json.loads(l) for l in open(ROOT / "results/zeroshot" / s / TUG / "predictions.jsonl", encoding="utf-8")]
    ge = [edit_counts(p["hyp"], p["ref"]) for p in greedy]
    gw = 100 * sum(e["word_edits"] for e in ge) / sum(e["ref_words"] for e in ge)
    ax.axhline(gw, color=color, ls="--", lw=0.7)
    rows.append({"set": name, "n": 0, "wer": gw, "lo": None, "hi": None})
ax.set_xlabel("N (candidates in the list)")
ax.set_ylabel("oracle WER (%)")
ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
save(fig, "f9_oracle_vs_n", pd.DataFrame(rows))
