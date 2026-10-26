"""
Reference noise by forced-aligner confidence on the verified DL Sprint eval clips: for each tercile of the clip's
mean word posterior, the share of clips the annotators corrected and the word edit rate original -> verified, with a
cluster-bootstrap 95% CI over recordings. Answers whether the WER rise in low-confidence clips (Figure 11) is
reference noise.

Inputs: results/labels/verified_references.csv (import_labels.py), data/splits_fa/eval.csv
Output: results/labels/noise_by_alignment.csv
  PYTHONPATH=src .venv/bin/python analysis/labeling/noise_by_alignment.py
"""

import sys

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text as N  # noqa: E402

V = pd.read_csv("results/labels/verified_references.csv")
V = V[(V.source == "dlsprint") & (V.status != "unusable")]
D = V.merge(pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig")[["utt_id", "mean_score"]], on="utt_id")
assert len(D) == len(V), "every verified clip must be in the eval manifest"
D["edits"] = [Levenshtein.distance(N(a).split(), N(b).split()) for a, b in zip(D.text_original, D.text_verified)]
D["words"] = [len(N(a).split()) for a in D.text_original]
D["tercile"] = pd.qcut(D.mean_score, 3, labels=["low", "mid", "high"])
rows = []
for t, g in D.groupby("tercile", observed=True):
    r = bootstrap_ratio(g.recording.tolist(), g.edits.tolist(), g.words.tolist(), n_boot=1000)
    rows.append({"tercile": t, "score_min": g.mean_score.min(), "score_max": g.mean_score.max(), "clips": len(g),
                 "corrected_share": (g.status == "corrected").mean(), "edit_rate": r["value"],
                 "ci_low": r["ci95"][0], "ci_high": r["ci95"][1]})
out = pd.DataFrame(rows)
out.to_csv("results/labels/noise_by_alignment.csv", index=False)
print(out.round(4).to_string(index=False))
