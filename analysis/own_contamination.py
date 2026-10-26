#!/usr/bin/env python3
"""
Is our own final model contaminated? Its training data is the M5-300 list (M1's DL Sprint list is a subset of it).
Checked against every test set:

  recordings   DL Sprint: no training clip comes from a dev or eval recording
  speakers     Common Voice (client id) and Kathbath (speaker id): no test speaker in the training list
  sentences    exact normalized sentence matches between each test set and the training text
  memorization the final model on DL Sprint eval clips whose reference the annotators corrected is closer to the
               corrected text than to the original (results/labels/memorization_check.csv)
  overlap      the final model's WER with and without the clips that share a sentence with the training text

Inputs: data/backup/modal_friend/manifests/matrix/m5-300.csv (training list, local), data/splits_fa/{train,dev,eval}.csv,
        data/external/{cv/test,fleurs/test,kathbath/valid}.csv, results/labels/memorization_check.csv
Output: results/contamination/own_model.json
  PYTHONPATH=src .venv/bin/python analysis/own_contamination.py
"""

import json
import sys

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text as N  # noqa: E402


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


train = read("data/backup/modal_friend/manifests/matrix/m5-300.csv")
m1 = read("data/splits_fa/train_clean.csv")
assert m1.utt_id.isin(train.utt_id).all(), "M1's list must be part of the M5-300 list"
tests = {"splits_fa-eval": read("data/splits_fa/eval.csv"), "fleurs-test": read("data/external/fleurs/test.csv"),
         "cv-test": read("data/external/cv/test.csv"), "kathbath-valid": read("data/external/kathbath/valid.csv")}
dev_eval = set(read("data/splits_fa/dev.csv").recording) | set(tests["splits_fa-eval"].recording)
dl = train[train.source == "train_clean"]
train_text = {N(str(t)) for t in train.text}

out = {"training_clips": len(train), "training_sources": train.source.value_counts().to_dict(),
       "dlsprint_training_clips_from_dev_or_eval_recordings": int(dl.group.isin(dev_eval).sum()),
       "cv_test_speakers_in_training": int(tests["cv-test"].group.astype(str).isin(
           set(train[train.source == "cv-train"].group.astype(str))).sum()),
       "kathbath_valid_speakers_in_training": int(tests["kathbath-valid"].group.astype(str).isin(
           set(train[train.source == "kathbath-train"].group.astype(str))).sum()),
       "fleurs_clips_in_training": int(train.source.str.contains("fleurs").sum()),
       "sentence_overlap": {}}
for s, d in tests.items():
    hits = [u for u, t in zip(d.utt_id, d.text) if N(str(t)) in train_text]
    out["sentence_overlap"][s] = {"clips": len(d), "clips_with_training_sentence": len(hits),
                                  "examples": [N(str(t)) for u, t in zip(d.utt_id, d.text) if u in set(hits)][:5]}
FINAL = "results/kaggle/finetuned/{}/soup_m1_full+m5_300+m5_300+m5_300_t4/predictions.jsonl"
out["final_model_wer_without_overlapping_clips"] = {}
for s, d in tests.items():
    drop = {u for u, t in zip(d.utt_id, d.text) if N(str(t)) in train_text}
    rows = [json.loads(l) for l in open(FINAL.format(s), encoding="utf-8")]
    def wer(rs):
        return 100 * sum(Levenshtein.distance(N(r["ref"]).split(), N(r["hyp"]).split()) for r in rs) / sum(len(N(r["ref"]).split()) for r in rs)
    out["final_model_wer_without_overlapping_clips"][s] = {"all": round(wer(rows), 3),
                                                            "without": round(wer([r for r in rows if r["utt_id"] not in drop]), 3)}
mem = read("results/labels/memorization_check.csv").set_index("system")
out["memorization_final_model"] = mem.loc["final model", ["closer_to_original", "closer_to_corrected"]].astype(int).to_dict()
open("results/contamination/own_model.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
