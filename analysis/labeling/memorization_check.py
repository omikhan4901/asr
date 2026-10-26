"""
Memorization check on the DL Sprint eval clips whose reference the annotators corrected: a model that recognizes the
speech should be closer to the corrected text; a model trained on these recordings' transcripts reproduces the
original errors and is closer to the uncorrected text. For each system: WER against both texts, and the number of
clips closer to each.

Inputs: results/labels/verified_references.csv, each system's predictions.jsonl on splits_fa-eval
Output: results/labels/memorization_check.csv
  PYTHONPATH=src .venv/bin/python analysis/labeling/memorization_check.py
"""

import json
import sys

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text as N  # noqa: E402

E = "results/{}/splits_fa-eval/{}"
SYSTEMS = {"zarifmahir21 (trained on DL Sprint 4.0 data)": E.format("kaggle/zeroshot", "zarifmahir21__whisper-medium-bangla_t4"),
           "Team Villagers (trained on DL Sprint 4.0 data)": E.format("kaggle/zeroshot", "teamvillagers__whisper-bengali-ct2_t4"),
           "final model": E.format("kaggle/finetuned", "soup_m1_full+m5_300+m5_300+m5_300_t4"),
           "M1 (DL Sprint train split only)": E.format("modal/finetuned", "m1_full"),
           "tugstugi zero-shot": E.format("modal/zeroshot", "tugstugi_a100"),
           "Hishab conformer large": E.format("zeroshot", "hishab__titu_stt_bn_conformer_large")}

V = pd.read_csv("results/labels/verified_references.csv")
V = V[(V.source == "dlsprint") & (V.status == "corrected")]
V = V[[N(a) != N(b) for a, b in zip(V.text_original, V.text_verified)]]
rows = []
for name, path in SYSTEMS.items():
    hyp = {r["utt_id"]: N(r["hyp"]).split() for r in map(json.loads, open(f"{path}/predictions.jsonl", encoding="utf-8"))}
    eo = ev = w = closer_o = closer_v = 0
    for u, o, v in zip(V.utt_id, V.text_original, V.text_verified):
        h, o, v = hyp[u], N(o).split(), N(v).split()
        do, dv = Levenshtein.distance(o, h), Levenshtein.distance(v, h)
        eo, ev, w = eo + do, ev + dv, w + len(v)
        closer_o, closer_v = closer_o + (do < dv), closer_v + (dv < do)
    rows.append({"system": name, "clips": len(V), "wer_vs_original": eo / w, "wer_vs_corrected": ev / w,
                 "closer_to_original": closer_o, "closer_to_corrected": closer_v})
out = pd.DataFrame(rows)
out.to_csv("results/labels/memorization_check.csv", index=False)
print(out.round(4).to_string(index=False))
