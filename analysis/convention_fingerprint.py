"""
E20: spelling-convention fingerprint of Bangla corpora and of model outputs.

Words are grouped into variant classes by bangla_asr.common.variants.fold
(ছিল/ছিলো, কি/কী, কোন/কোনো, ...). A class is kept if at least two surface
forms each occur >= --min_form times over all sources. For every source
(reference texts of a corpus, or the hypotheses of a model on a test set) we
count each form, giving a per-class form distribution: the source's
convention fingerprint.

Distance between two sources = token-weighted mean over shared classes of
the total-variation distance between their form distributions (0 = same
conventions, 1 = always the opposite form). Only classes with >= --min_class
tokens in both sources count. The model-vs-reference distance on a test set
is then compared with that model's variant share of WER (E21).

Outputs in --out_dir: classes.csv (per class and source: count and share of
the overall majority form), distance.csv (source x source), model_vs_ref.csv.

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/convention_fingerprint.py
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import read_jsonl  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import fold  # noqa: E402

# Reference corpora: name -> list of (csv/tsv path, text column)
CORPORA = {
    "dlsprint": [("data/splits_fa/train.csv", "text"), ("data/splits_fa/dev.csv", "text"),
                 ("data/splits_fa/eval.csv", "text")],
    "fleurs": [("data/external/fleurs/test.csv", "text")],
    "kathbath": [("data/external/kathbath/valid.csv", "text")],
    "commonvoice": [("data/CommonVoiceCorpus/bn/train.tsv", "sentence"),
                    ("data/CommonVoiceCorpus/bn/dev.tsv", "sentence"),
                    ("data/CommonVoiceCorpus/bn/test.tsv", "sentence")],
}
# Zero-shot result folder -> reference corpus it is scored against
SET_TO_CORPUS = {"splits_fa-eval": "dlsprint", "fleurs-test": "fleurs", "kathbath-valid": "kathbath",
                 "cv-test": "commonvoice"}


def read_texts(path, col):
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, usecols=[col])
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=[col])
    return df[col].fillna("").tolist()


def count_words(texts):
    c = Counter()
    for t in texts:
        c.update(normalize_bangla_text(t).split())
    return c


def tv_distance(a: Counter, b: Counter):
    na, nb = sum(a.values()), sum(b.values())
    forms = set(a) | set(b)
    return 0.5 * sum(abs(a[f] / na - b[f] / nb) for f in forms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/fingerprint")
    p.add_argument("--min_form", type=int, default=20, help="min count of a form (all sources pooled)")
    p.add_argument("--min_class", type=int, default=10, help="min class tokens in a source to compare it")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = [f for v in CORPORA.values() for f, _ in v if Path(f).exists()]
    preds = sorted(Path("results/zeroshot").glob("*/*/predictions.jsonl"))
    with RunRecord(out / "run_info.json", inputs=inputs + preds) as rec:
        sources = {}  # name -> Counter of words
        for name, files in CORPORA.items():
            texts = [t for f, col in files if Path(f).exists() for t in read_texts(f, col)]
            if texts:
                sources[f"ref:{name}"] = count_words(texts)
        for pp in preds:
            corpus = SET_TO_CORPUS.get(pp.parent.parent.name)
            if corpus:
                sources[f"hyp:{pp.parent.name}@{corpus}"] = count_words(r["hyp"] for r in read_jsonl(pp))

        # Variant classes from the reference corpora only (model outputs do not define classes).
        pooled = Counter()
        for n, c in sources.items():
            if n.startswith("ref:"):
                pooled.update(c)
        by_key = defaultdict(list)
        for w, n in pooled.items():
            if n >= args.min_form:
                by_key[fold(w)].append(w)
        classes = {k: sorted(v, key=lambda w: -pooled[w]) for k, v in by_key.items() if len(v) >= 2}
        print(f"{len(sources)} sources, {len(classes)} variant classes")

        fp = {n: {k: Counter({w: c[w] for w in forms if c[w]}) for k, forms in classes.items()}
              for n, c in sources.items()}

        rows = []
        for k, forms in sorted(classes.items(), key=lambda kv: -sum(pooled[w] for w in kv[1])):
            row = {"class": k, "forms": " / ".join(forms), "majority_form": forms[0],
                   "pooled_tokens": sum(pooled[w] for w in forms)}
            for n in sources:
                cnt = fp[n][k]
                tot = sum(cnt.values())
                row[f"{n}|n"] = tot
                row[f"{n}|majority_share"] = round(cnt[forms[0]] / tot, 3) if tot else None
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "classes.csv", index=False, encoding="utf-8-sig")

        def distance(a, b):
            num = den = 0.0
            shared = 0
            for k in classes:
                ca, cb = fp[a][k], fp[b][k]
                na, nb = sum(ca.values()), sum(cb.values())
                if na >= args.min_class and nb >= args.min_class:
                    w = min(na, nb)
                    num += w * tv_distance(ca, cb)
                    den += w
                    shared += 1
            return (num / den if den else float("nan")), shared

        names = list(sources)
        dist = pd.DataFrame(0.0, index=names, columns=names)
        for a, b in combinations(names, 2):
            dist.loc[a, b] = dist.loc[b, a] = round(distance(a, b)[0], 4)
        dist.to_csv(out / "distance.csv", encoding="utf-8-sig")

        # Model output conventions vs the references it is scored against, next to E21.
        e21 = {}
        if Path("results/error_types/summary.csv").exists():
            s = pd.read_csv("results/error_types/summary.csv", encoding="utf-8-sig")
            e21 = {(r["model"], SET_TO_CORPUS.get(r["set"])): r["variant_share%"] for _, r in s.iterrows()}
        mv = []
        for n in names:
            if n.startswith("hyp:"):
                model, corpus = n[4:].rsplit("@", 1)
                d, shared = distance(n, f"ref:{corpus}")
                mv.append({"model": model, "corpus": corpus, "tv_distance": round(d, 4), "classes_compared": shared,
                           "variant_share_of_wer%": e21.get((model, corpus))})
        pd.DataFrame(mv).to_csv(out / "model_vs_ref.csv", index=False, encoding="utf-8-sig")
        rec.outputs = [out / "classes.csv", out / "distance.csv", out / "model_vs_ref.csv"]

    refs = [n for n in names if n.startswith("ref:")]
    print(dist.loc[refs, refs].to_string())
    print(pd.DataFrame(mv).to_string(index=False))


if __name__ == "__main__":
    main()
