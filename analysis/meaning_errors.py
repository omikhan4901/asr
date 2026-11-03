#!/usr/bin/env python3
"""
Errors on meaning-carrying words (PLAN §12, E27; pre-registered in PROGRESS §79, amendment §84).

Next to WER (unchanged), per system and test set:
  CWER       substituted + deleted reference content words / reference content words
             (content word = not one of the 200 most frequent DL Sprint train words, results/retrieval/stopwords.csv)
  FWER       the same on the function (stop) words
  ref_rate   (S + D) / N: WER without insertions, the fair comparison for CWER and FWER
  IDF-WER    (IDF of substituted/deleted reference words + IDF of inserted words) / IDF of all reference words;
             IDF(w) = ln((N+1)/(df+1)) + 1 over DL Sprint train clips
  pairs      substitutions whose (reference, hypothesis) pair the native annotators labelled "different" (meaning
             changes) or "spelling" (meaning kept), per 1,000 reference words (results/meaning_errors/pair_list.csv)

Same normalizer and alignment as the WER (edit count asserted per clip); 95% cluster-bootstrap CIs over the
predictions' group field (recordings / speakers), paired across systems.
Outputs: results/meaning_errors/{pair_list.csv, summary.csv, predictions.csv, run_info.json}
  PYTHONPATH=src .venv/bin/python analysis/meaning_errors.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import align_words  # noqa: E402
from retrieval import FINAL, SETS, SYSTEMS, ZERO  # noqa: E402
from spelling_keys import pair_labels  # noqa: E402

STOPWORDS = "results/retrieval/stopwords.csv"
TRAIN_TEXT = "data/splits_fa/train_clean.csv"
OUT = Path("results/meaning_errors")
N_BOOT, SEED = 2000, 0
COUNTS = ["N", "E", "ins", "Nc", "Ec", "Nf", "Ef", "idf_ref", "idf_err", "diff", "spell"]


def norm(text):
    return normalize_bangla_text(str(text)).split()


def per_clip(path: Path, stop, idf, idf_unseen, diff_pairs, spell_pairs) -> pd.DataFrame:
    rows = []
    for line in open(path / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ref, hyp = norm(r["ref"]), norm(r["hyp"])
        ops = align_words(ref, hyp)
        edits = [o for o in ops if o.op != "equal"]
        assert len(edits) == Levenshtein.distance(ref, hyp), r["utt_id"]
        c = dict.fromkeys(COUNTS, 0.0)
        c["N"] = len(ref)
        c["E"] = len(edits)
        c["idf_ref"] = sum(idf.get(w, idf_unseen) for w in ref)
        for w in ref:
            c["Nf" if w in stop else "Nc"] += 1
        for o in edits:
            if o.op == "insert":
                c["ins"] += 1
                c["idf_err"] += idf.get(o.hyp_token, idf_unseen)
                continue
            c["Ef" if o.ref_token in stop else "Ec"] += 1
            c["idf_err"] += idf.get(o.ref_token, idf_unseen)
            if o.op == "substitute":
                c["diff"] += (o.ref_token, o.hyp_token) in diff_pairs
                c["spell"] += (o.ref_token, o.hyp_token) in spell_pairs
        rows.append({"utt_id": r["utt_id"], "group": str(r["group"]), **c})
    return pd.DataFrame(rows)


def metrics(t: np.ndarray) -> dict:
    """t: summed COUNTS (array in COUNTS order) -> rates in %."""
    d = dict(zip(COUNTS, t))
    return {"WER": 100 * d["E"] / d["N"], "ref_rate": 100 * (d["E"] - d["ins"]) / d["N"],
            "CWER": 100 * d["Ec"] / d["Nc"], "FWER": 100 * d["Ef"] / d["Nf"],
            "IDF_WER": 100 * d["idf_err"] / d["idf_ref"],
            "diff_per_1k": 1000 * d["diff"] / d["N"], "spell_per_1k": 1000 * d["spell"] / d["N"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [STOPWORDS, TRAIN_TEXT, "results/labels/pair_labels.csv"]
    for s in SETS:
        inputs += [Path(p.format(s)) / "predictions.jsonl" for p in SYSTEMS.values() if Path(p.format(s)).exists()]
    with RunRecord(OUT / "run_info.json", inputs=inputs) as rec:
        stop = set(pd.read_csv(STOPWORDS, encoding="utf-8-sig")["stopword"])
        train = pd.read_csv(TRAIN_TEXT, encoding="utf-8-sig", usecols=["text"])["text"]
        df = Counter(w for t in train for w in set(norm(t)))
        n = len(train)
        idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        idf_unseen = math.log(n + 1) + 1

        P = pair_labels()
        P["ref_word"] = P["ref_word"].map(lambda x: normalize_bangla_text(str(x)))
        P["hyp_word"] = P["hyp_word"].map(lambda x: normalize_bangla_text(str(x)))
        P[["ref_word", "hyp_word", "label", "rule_type", "stratum"]].to_csv(OUT / "pair_list.csv", index=False,
                                                                            encoding="utf-8-sig")
        diff_pairs = set(zip(*P[P["label"] == "different"][["ref_word", "hyp_word"]].T.values))
        spell_pairs = set(zip(*P[P["label"] == "spelling"][["ref_word", "hyp_word"]].T.values))

        summ, preds = [], []
        for s in SETS:
            clips = {name: per_clip(Path(p.format(s)), stop, idf, idf_unseen, diff_pairs, spell_pairs)
                     for name, p in SYSTEMS.items() if Path(p.format(s)).exists()}
            base = clips[FINAL]
            groups = base["group"].to_numpy()
            uniq = {g: k for k, g in enumerate(dict.fromkeys(groups))}
            gi = np.array([uniq[g] for g in groups])
            G = len(uniq)
            samp = np.random.default_rng(SEED).integers(0, G, size=(N_BOOT, G))
            weights = np.stack([np.bincount(row, minlength=G) for row in samp])  # times each group is drawn
            boot = {}
            for name, d in clips.items():
                d = d.set_index("utt_id").loc[base["utt_id"]]
                per_group = np.zeros((G, len(COUNTS)))
                np.add.at(per_group, gi, d[COUNTS].to_numpy())
                point = metrics(per_group.sum(axis=0))
                reps = [metrics(t) for t in weights @ per_group]
                boot[name] = (point, reps)
                row = {"set": s, "system": name, "clips": len(d), "ref_words": int(per_group[:, 0].sum()),
                       "content_share": 100 * per_group[:, COUNTS.index("Nc")].sum() / per_group[:, 0].sum()}
                for k, v in point.items():
                    vals = [r[k] for r in reps]
                    row.update({k: v, f"{k}_lo": np.percentile(vals, 2.5), f"{k}_hi": np.percentile(vals, 97.5)})
                summ.append(row)
            fp, fr = boot[FINAL]
            zp, zr = boot[ZERO]
            # P27a: final CWER - final WER (and vs the insertion-free rate)
            for ref_metric in ("WER", "ref_rate"):
                v = [r["CWER"] - r[ref_metric] for r in fr]
                preds.append({"prediction": "P27a", "set": s, "contrast": f"final CWER - final {ref_metric}",
                              "diff": fp["CWER"] - fp[ref_metric], "lo": np.percentile(v, 2.5),
                              "hi": np.percentile(v, 97.5)})
            # P27b: relative fine-tuning gain on CWER minus on WER
            rel = lambda z, f, k: (z[k] - f[k]) / z[k]  # noqa: E731
            v = [rel(z, f, "CWER") - rel(z, f, "WER") for z, f in zip(zr, fr)]
            preds.append({"prediction": "P27b", "set": s, "contrast": "relative gain CWER - relative gain WER",
                          "diff": rel(zp, fp, "CWER") - rel(zp, fp, "WER"), "lo": np.percentile(v, 2.5),
                          "hi": np.percentile(v, 97.5)})
            print(f"{s}: done", flush=True)
        S = pd.DataFrame(summ)
        S.to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
        rec.outputs = [OUT / "pair_list.csv", OUT / "summary.csv", OUT / "predictions.csv"]
        cols = ["WER", "ref_rate", "CWER", "FWER", "IDF_WER", "diff_per_1k", "spell_per_1k", "content_share"]
        print(S.set_index(["set", "system"])[cols].round(2).to_string())
        print(pd.DataFrame(preds).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
