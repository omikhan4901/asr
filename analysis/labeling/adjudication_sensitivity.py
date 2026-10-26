"""
Sensitivity of the verified-reference WER to the adjudication: the settled reference disagreements were decided with
two recognizers' outputs as evidence (one of them the final model), so recompute each system's WER on the verified
clips with the other annotator's text on every settled clip. A difference near 0 for the final model, and no larger
than for the baselines, means the adjudication does not favour it.

Inputs: results/labels/adjudication.json, results/labels/verified_references.csv, the snapshot named in the
        adjudication file, each system's predictions.jsonl
Output: results/labels/adjudication_sensitivity.csv
  PYTHONPATH=src .venv/bin/python analysis/labeling/adjudication_sensitivity.py
"""

import json
import sys
from collections import defaultdict

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text as N  # noqa: E402

SETS = {"dlsprint": "splits_fa-eval", "fleurs": "fleurs-test", "kathbath": "kathbath-valid", "commonvoice": "cv-test"}
SYSTEMS = {"final model": "results/kaggle/finetuned/{}/soup_m1_full+m5_300+m5_300+m5_300_t4",
           "tugstugi zero-shot": "results/modal/zeroshot/{}/tugstugi_a100",
           "Hishab conformer large": "results/zeroshot/{}/hishab__titu_stt_bn_conformer_large",
           "IndicConformer CTC": "results/kaggle/zeroshot/{}/ai4bharat__indic-conformer-600m-multilingual_ctc_t4"}

adj = json.load(open("results/labels/adjudication.json", encoding="utf-8"))
snap = json.load(open(adj["snapshot"], encoding="utf-8"))
items = {i["id"]: i for i in snap["items"]}
texts = defaultdict(dict)
for l in snap["labels"]:
    if l["status"] == "done" and items[l["item_id"]]["task"] == "ref":
        texts[l["item_id"]][l["annotator"]] = l["data"]["text"]
other = {}
for r in adj["resolutions"]:
    if r["task"] == "ref":
        (alt,) = [a for a in texts[r["item_id"]] if a != r["chosen"]]
        other[items[r["item_id"]]["meta"]["utt_id"]] = texts[r["item_id"]][alt]

V = pd.read_csv("results/labels/verified_references.csv")
rows = []
for src, s in SETS.items():
    chosen = V[(V.source == src) & (V.status != "unusable")].set_index("utt_id")["text_verified"].to_dict()
    flipped = {u: other.get(u, t) for u, t in chosen.items()}
    for name, path in SYSTEMS.items():
        hyp = {r["utt_id"]: r["hyp"] for r in map(json.loads, open(path.format(s) + "/predictions.jsonl", encoding="utf-8"))}

        def wer(refs):
            return (sum(Levenshtein.distance(N(refs[u]).split(), N(hyp[u]).split()) for u in refs)
                    / sum(len(N(refs[u]).split()) for u in refs))

        rows.append({"set": src, "system": name, "settled_clips": sum(u in other for u in chosen),
                     "wer_adjudicated": wer(chosen), "wer_other_answer": wer(flipped)})
out = pd.DataFrame(rows)
out["difference"] = out.wer_adjudicated - out.wer_other_answer
out.to_csv("results/labels/adjudication_sensitivity.csv", index=False)
print((out.set_index(["set", "system"]) * [1, 100, 100, 100]).round(2).to_string())
