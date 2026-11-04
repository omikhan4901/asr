#!/usr/bin/env python3
"""
Retrieval over ASR transcripts (PLAN §12, E26; pre-registered in PROGRESS §79, amendment §82).

Does a better recognizer, and does spelling-insensitive matching, help someone search the transcripts?

  Keyword search (all four test sets): a query is one reference word (not a stopword) that occurs in 2-50 clips of
      that set's original references; up to 500 terms per set, sampled with seed 0 and saved by `terms` BEFORE any
      system is scored. Relevant clips = clips whose original reference contains the term.
  Sentence search (FLEURS test only): a query is one of the 349 normalized reference sentences; relevant clips =
      every reading of that sentence.
  Ranking: BM25 (k1 1.2, b 0.75) over each system's transcripts. Metrics: MAP (primary), recall@10 (keywords),
      P@1 (sentences). Docs with score 0 are not retrieved; ties broken by clip order.
  Matching conditions, applied to query and transcripts alike: exact words, fold(), the deep key and UzZaman & Khan
      2004 (analysis/spelling_keys.py), character 3-grams inside words, and family() (one ending stripped, then fold).
  CIs: 2,000 bootstrap resamples of queries (terms or sentences), seed 0, paired across systems and conditions.
  `--relevance family` decides relevance and matching by word family (one inflectional ending stripped, then fold) and
      skips the sentence task.

Inputs: saved predictions.jsonl only (no inference). Stopwords: the 200 most frequent words of DL Sprint train
references (data/splits_fa/train_clean.csv).
Outputs: results/retrieval/{terms_<set>.csv, per_query.csv, summary.csv, predictions.csv, run_info*.json}
  PYTHONPATH=src .venv/bin/python analysis/retrieval.py terms
  PYTHONPATH=src .venv/bin/python analysis/retrieval.py score
  PYTHONPATH=src .venv/bin/python analysis/retrieval.py score --relevance family --conditions exact,fold,family --out_dir results/retrieval_family
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.stem import family  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import fold  # noqa: E402
from spelling_keys import deep, uzzaman04  # noqa: E402

SETS = ["splits_fa-eval", "fleurs-test", "cv-test", "kathbath-valid"]
FINAL, ZERO = "final", "zero-shot"
SYSTEMS = {  # label -> folder pattern ({} = test set); PROGRESS §79
    ZERO: "results/modal/zeroshot/{}/tugstugi_a100",
    "list correction": "results/list_correction/{}/rewrite",  # no CV output
    "M1": "results/modal/finetuned/{}/m1_full",
    "Hishab": "results/zeroshot/{}/hishab__titu_stt_bn_conformer_large",
    "IndicConformer CTC": "results/kaggle/zeroshot/{}/ai4bharat__indic-conformer-600m-multilingual_ctc_t4",
    FINAL: "results/kaggle/finetuned/{}/soup_m1_full+m5_300+m5_300+m5_300_t4",
}
TRAIN_TEXT = "data/splits_fa/train_clean.csv"
OUT = Path("results/retrieval")
N_STOP, DF_MIN, DF_MAX, N_TERMS = 200, 2, 50, 500
K1, B = 1.2, 0.75
N_BOOT, SEED = 2000, 0


def char3(word: str):
    w = f"<{word}>"
    return [w[i:i + 3] for i in range(len(w) - 2)] if len(w) > 3 else [w]


CONDITIONS = {
    "exact": lambda w: [w],
    "fold": lambda w: [fold(w)],
    "deep": lambda w: [deep(w)],
    "uzzaman04": lambda w: [uzzaman04(w)],
    "family": lambda w: [family(w)],
    "char3": char3,
}


def norm(text: str):
    return normalize_bangla_text(str(text)).split()


def read_preds(path: Path):
    return [json.loads(line) for line in open(path / "predictions.jsonl", encoding="utf-8")]


def references(s: str):
    """utt_id -> normalized reference words, in the final model's clip order (the canonical clip list)."""
    rows = read_preds(Path(SYSTEMS[FINAL].format(s)))
    return {r["utt_id"]: norm(r["ref"]) for r in rows}


def stopwords():
    df = pd.read_csv(TRAIN_TEXT, encoding="utf-8-sig", usecols=["text"])
    return {w for w, _ in Counter(w for t in df["text"] for w in norm(t)).most_common(N_STOP)}


def terms(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with RunRecord(OUT / "run_info_terms.json", inputs=[TRAIN_TEXT, *(Path(SYSTEMS[FINAL].format(s)) / "predictions.jsonl" for s in SETS)]) as rec:
        stop = stopwords()
        pd.Series(sorted(stop), name="stopword").to_csv(OUT / "stopwords.csv", index=False, encoding="utf-8-sig")
        rec.outputs = [OUT / "stopwords.csv"]
        for s in SETS:
            path = OUT / f"terms_{s}.csv"
            if path.exists() and not args.force:
                raise SystemExit(f"{path} exists: term lists are frozen (use --force only before any score)")
            refs = references(s)
            df = Counter(w for ws in refs.values() for w in set(ws))
            cand = sorted(w for w, c in df.items() if DF_MIN <= c <= DF_MAX and w not in stop)
            rng = np.random.default_rng(SEED)
            pick = sorted(rng.choice(cand, size=min(N_TERMS, len(cand)), replace=False).tolist())
            pd.DataFrame({"term": pick, "df": [df[w] for w in pick]}).to_csv(path, index=False, encoding="utf-8-sig")
            rec.outputs.append(path)
            print(f"{s}: {len(refs)} clips, {len(cand)} candidate terms, {len(pick)} sampled")


class BM25:
    def __init__(self, docs):  # docs: list of token lists
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if self.n else 0.0
        self.post = defaultdict(dict)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t][i] = c

    def scores(self, query_tokens):
        sc = np.zeros(self.n)
        for t in set(query_tokens):
            p = self.post.get(t)
            if not p:
                continue
            idf = math.log((self.n - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            idx = np.fromiter(p.keys(), dtype=int)
            tf = np.fromiter(p.values(), dtype=float)
            sc[idx] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.len[idx] / self.avg))
        return sc


def rank_metrics(scores: np.ndarray, relevant: np.ndarray):
    """AP over the retrieved list (score > 0), recall@10, P@1. Ties: clip order (stable sort)."""
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] > 0]
    hits = relevant[order]
    n_rel = relevant.sum()
    ap = (np.cumsum(hits) / np.arange(1, len(hits) + 1))[hits].sum() / n_rel if len(hits) else 0.0
    return ap, hits[:10].sum() / n_rel, float(hits[:1].sum()) if len(hits) else 0.0


def keyed(words, cond):
    f = CONDITIONS[cond]
    return [k for w in words for k in f(w)]


def score(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    inputs = [OUT / f"terms_{s}.csv" for s in SETS]
    for s in SETS:
        inputs += [Path(p.format(s)) / "predictions.jsonl" for p in SYSTEMS.values() if Path(p.format(s)).exists()]
    with RunRecord(out / "run_info.json", inputs=inputs) as rec:
        rows = []
        for s in SETS:
            refs = references(s)
            ids = list(refs)
            pos = {u: i for i, u in enumerate(ids)}
            ref_sets = [set(refs[u]) for u in ids]
            T = pd.read_csv(OUT / f"terms_{s}.csv", encoding="utf-8-sig")
            if args.relevance == "family":  # family keys computed once per clip, not once per term
                ref_fams = [{family(w) for w in rs} for rs in ref_sets]
                kw_rel = {t: np.array([family(t) in fs for fs in ref_fams]) for t in T["term"]}
            else:
                kw_rel = {t: np.array([t in rs for rs in ref_sets]) for t in T["term"]}
            sent_rel = {}
            if s == "fleurs-test" and args.relevance != "family":
                by_sent = defaultdict(list)
                for u in ids:
                    by_sent[" ".join(refs[u])].append(pos[u])
                for q, members in by_sent.items():
                    rel = np.zeros(len(ids), dtype=bool)
                    rel[members] = True
                    sent_rel[q] = rel
            for sys_name, pat in SYSTEMS.items():
                path = Path(pat.format(s))
                if not path.exists():
                    continue
                hyp = {r["utt_id"]: norm(r["hyp"]) for r in read_preds(path)}
                missing = [u for u in ids if u not in hyp]
                if missing:
                    raise SystemExit(f"{path}: {len(missing)} of {len(ids)} clips missing")
                for cond in conds:
                    index = BM25([keyed(hyp[u], cond) for u in ids])
                    for t, rel in kw_rel.items():
                        ap, r10, _ = rank_metrics(index.scores(keyed([t], cond)), rel)
                        rows.append({"set": s, "task": "keyword", "query": t, "system": sys_name, "condition": cond,
                                     "ap": ap, "recall10": r10, "p1": np.nan})
                    for q, rel in sent_rel.items():
                        ap, _, p1 = rank_metrics(index.scores(keyed(q.split(), cond)), rel)
                        rows.append({"set": s, "task": "sentence", "query": q, "system": sys_name, "condition": cond,
                                     "ap": ap, "recall10": np.nan, "p1": p1})
                print(f"{s} {sys_name}: done", flush=True)
        Q = pd.DataFrame(rows)
        Q.to_csv(out / "per_query.csv", index=False, encoding="utf-8-sig")
        summ, preds = summarize(Q, args.relevance)
        summ.to_csv(out / "summary.csv", index=False)
        preds.to_csv(out / "predictions.csv", index=False)
        rec.outputs = [out / "per_query.csv", out / "summary.csv", out / "predictions.csv"]
        print(summ[summ["task"] == "keyword"].pivot_table(index=["set", "system"], columns="condition", values="map")
              .round(1).to_string())
        print(preds.to_string(index=False))


def boot_means(Q: pd.DataFrame, s: str, task: str, rng):
    """Per (system, condition): MAP and its bootstrap replicates, resampling the same queries for all of them."""
    sub = Q[(Q["set"] == s) & (Q["task"] == task)]
    M = sub.pivot_table(index="query", columns=["system", "condition"], values="ap")
    idx = rng.integers(0, len(M), size=(N_BOOT, len(M)))
    vals = M.to_numpy()
    return M.columns, 100 * vals.mean(axis=0), 100 * vals[idx].mean(axis=1)  # MAP in points (0-100)


def summarize(Q: pd.DataFrame, relevance: str = "exact"):
    rng = np.random.default_rng(SEED)
    summ, reps = [], {}
    for task in ("keyword", "sentence"):
        for s in Q.loc[Q["task"] == task, "set"].unique():
            cols, point, boot = boot_means(Q, s, task, rng)
            for j, (sysn, cond) in enumerate(cols):
                sub = Q[(Q["set"] == s) & (Q["task"] == task) & (Q["system"] == sysn) & (Q["condition"] == cond)]
                summ.append({"set": s, "task": task, "system": sysn, "condition": cond, "queries": len(sub),
                             "map": point[j], "map_lo": np.percentile(boot[:, j], 2.5),
                             "map_hi": np.percentile(boot[:, j], 97.5),
                             "recall10": 100 * sub["recall10"].mean(), "p1": 100 * sub["p1"].mean()})
                reps[(task, s, sysn, cond)] = (point[j], boot[:, j])
    summ = pd.DataFrame(summ)

    def diff(name, a, b, sets):  # a - b, averaged over sets (each set resampled independently)
        pa = np.mean([reps[("keyword", s, *a)][0] - reps[("keyword", s, *b)][0] for s in sets])
        ba = np.mean([reps[("keyword", s, *a)][1] - reps[("keyword", s, *b)][1] for s in sets], axis=0)
        return {"contrast": name, "sets": "+".join(sets), "diff": pa,
                "lo": np.percentile(ba, 2.5), "hi": np.percentile(ba, 97.5)}

    def did(name, sysa, sysb, sets, cond="fold"):  # (cond - exact) of sysa minus the same for sysb
        def gain(s, sysn):
            pa, ba = reps[("keyword", s, sysn, cond)]
            pb, bb = reps[("keyword", s, sysn, "exact")]
            return pa - pb, ba - bb
        p = np.mean([gain(s, sysa)[0] - gain(s, sysb)[0] for s in sets])
        b = np.mean([gain(s, sysa)[1] - gain(s, sysb)[1] for s in sets], axis=0)
        return {"contrast": name, "sets": "+".join(sets), "diff": p,
                "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)}

    pr = []
    if relevance == "family":
        for s in SETS:
            pr.append({"prediction": "P28a", **diff("final - zero-shot, family", (FINAL, "family"), (ZERO, "family"), [s])})
        for sets in (["splits_fa-eval"], SETS):
            pr.append({"prediction": "P28b", **did("family gain: zero-shot - final", ZERO, FINAL, sets, "family")})
        for s in SETS:
            pr.append({"prediction": "P28c", **diff("final: family - exact", (FINAL, "family"), (FINAL, "exact"), [s])})
    else:
        for s in SETS:
            pr.append({"prediction": "P26a", **diff("final - zero-shot, exact", (FINAL, "exact"), (ZERO, "exact"), [s])})
        for sets in (["splits_fa-eval"], SETS):
            pr.append({"prediction": "P26b", **did("fold gain: zero-shot - final", ZERO, FINAL, sets)})
        for s in SETS:
            pr.append({"prediction": "P26c", **diff("final: fold - exact", (FINAL, "fold"), (FINAL, "exact"), [s])})
        for s in SETS:  # P26d: rank of the final model among the systems by exact keyword MAP
            k = summ[(summ["task"] == "keyword") & (summ["set"] == s) & (summ["condition"] == "exact")]
            k = k.sort_values("map", ascending=False).reset_index(drop=True)
            best_other = k[k["system"] != FINAL].iloc[0]
            pr.append({"prediction": "P26d", "contrast": f"final rank by exact MAP (best other: {best_other['system']})",
                       "sets": s, "diff": float(k.index[k["system"] == FINAL][0] + 1), "lo": np.nan, "hi": np.nan})
            pr.append({"prediction": "P26d", **diff(f"final - {best_other['system']}, exact", (FINAL, "exact"),
                                                    (best_other["system"], "exact"), [s])})
    return summ, pd.DataFrame(pr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("terms")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=terms)
    c = sub.add_parser("score")
    c.add_argument("--relevance", choices=("exact", "family"), default="exact")
    c.add_argument("--out_dir", default="results/retrieval")
    c.add_argument("--conditions", default="exact,fold,deep,uzzaman04,char3",
                   help="default = E26's five conditions, so the default run reproduces results/retrieval exactly")
    c.set_defaults(func=score)
    a = ap.parse_args()
    a.func(a)
