#!/usr/bin/env python3
"""
Keyword detection over ASR transcripts (pre-registered in PROGRESS §99).

One-word BM25 ranking mostly reduces to whether the word was found. Detection metrics measure that
directly, next to MAP: for each frozen term, whether the transcript contains the term's key.

ATWV is not computed: it needs word timestamps and a false-alarm weight calibrated in seconds of
speech, which clip-level data does not have.

Three settings (relevance, then matching):
  exact    relevant if the reference word set contains the term; detection with CONDITIONS["exact"]
  fold     relevance is exact; detection with CONDITIONS["fold"]
  family   relevant if any reference word has family(word) == family(term); detection with
           CONDITIONS["family"]

Outputs: results/keyword_detection/{summary.csv, contrasts.csv, run_info.json}
  PYTHONPATH=src .venv/bin/python analysis/keyword_detection.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.stem import family  # noqa: E402
from retrieval import (  # noqa: E402
    CONDITIONS, FINAL, SETS, SYSTEMS, ZERO, norm, read_preds, references,
)

OUT = Path("results/keyword_detection")
TERMS = Path("results/retrieval")
N_BOOT, SEED = 2000, 0
# name, relevance key, matching key
SETTINGS = (
    ("exact", "exact", "exact"),
    ("fold", "exact", "fold"),
    ("family", "family", "family"),
)
SUMMARY_COLS = ["set", "system", "setting", "terms", "macro_recall", "macro_precision", "f1",
                "micro_recall", "micro_precision"]
CONTRAST_COLS = ["prediction", "contrast", "sets", "diff", "ci_lo", "ci_hi", "significant"]


def keyed(words, cond):
    f = CONDITIONS[cond]
    return [k for w in words for k in f(w)]


def defined_mean(vals):
    """Mean over the last axis, ignoring NaN. A row that is all NaN is 0 (no detected term)."""
    vals = np.atleast_2d(np.asarray(vals, dtype=float))
    defined = ~np.isnan(vals)
    counts = defined.sum(axis=1)
    sums = np.where(defined, vals, 0.0).sum(axis=1)
    out = np.zeros(vals.shape[0], dtype=float)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def f1_of(recall_mean, precision_mean):
    """F1 of the two macro means. Precision 0 and recall 0 -> 0."""
    r = np.asarray(recall_mean, dtype=float)
    p = np.asarray(precision_mean, dtype=float)
    total = r + p
    if total.ndim == 0:
        return float(2 * p * r / total) if total > 0 else 0.0
    out = np.zeros(total.shape, dtype=float)
    ok = total > 0
    out[ok] = 2 * p[ok] * r[ok] / total[ok]
    return out


def relevance(ref_words, ids, terms, kind):
    """Clip x term truth. exact: the term itself. family: any reference word in the term's family."""
    rel = np.zeros((len(terms), len(ids)), dtype=bool)
    if kind == "exact":
        index = {t: j for j, t in enumerate(terms)}
        for i, u in enumerate(ids):
            for w in set(ref_words[u]):
                j = index.get(w)
                if j is not None:
                    rel[j, i] = True
        return rel
    index = defaultdict(list)
    for j, t in enumerate(terms):
        index[family(t)].append(j)
    for i, u in enumerate(ids):
        for key in {family(w) for w in ref_words[u]}:
            for j in index.get(key, ()):
                rel[j, i] = True
    return rel


def detection(hyp_words, ids, terms, cond):
    """True where the clip's key set contains a key of the term. Key sets are built once."""
    det = np.zeros((len(terms), len(ids)), dtype=bool)
    index = defaultdict(list)
    for j, term in enumerate(terms):
        for k in CONDITIONS[cond](term):
            index[k].append(j)
    for i, u in enumerate(ids):
        for k in set(keyed(hyp_words[u], cond)):
            js = index.get(k)
            if js:
                det[js, i] = True
    return det


def rates(det, rel):
    """Per term tp, fp, fn and recall / precision. Precision is NaN when the term has no detection
    or no relevant clip (those terms are skipped)."""
    tp = (det & rel).sum(axis=1).astype(float)
    fp = (det & ~rel).sum(axis=1).astype(float)
    fn = (~det & rel).sum(axis=1).astype(float)
    n_rel = tp + fn
    detected = tp + fp
    recall = np.full(tp.shape, np.nan)
    precision = np.full(tp.shape, np.nan)
    np.divide(tp, n_rel, out=recall, where=n_rel > 0)
    np.divide(tp, detected, out=precision, where=(n_rel > 0) & (detected > 0))
    return tp, fp, fn, recall, precision


def micro(tp, fp, fn):
    n_rel = tp + fn
    valid = n_rel > 0
    tp_s = float(tp[valid].sum())
    rel_s = float(n_rel[valid].sum())
    det_s = float((tp + fp)[valid].sum())
    return int(valid.sum()), (tp_s / rel_s if rel_s else 0.0), (tp_s / det_s if det_s else 0.0)


def contrast_row(prediction, name, sets, point, boot):
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"prediction": prediction, "contrast": name, "sets": sets, "diff": float(point),
            "ci_lo": lo, "ci_hi": hi, "significant": bool(not (lo <= 0 <= hi))}


def contrasts_of(point, boot):
    rows = []
    for s in SETS:
        rows.append(contrast_row(
            "P33a", "final - zero-shot, exact", s,
            point[(s, FINAL, "exact")] - point[(s, ZERO, "exact")],
            boot[(s, FINAL, "exact")] - boot[(s, ZERO, "exact")]))

    def gain(s, sys_name, store):
        return store[(s, sys_name, "fold")] - store[(s, sys_name, "exact")]

    for sets in (["splits_fa-eval"], SETS):
        pts, boots = [], []
        for s in sets:
            pts.append(gain(s, ZERO, point) - gain(s, FINAL, point))
            boots.append(gain(s, ZERO, boot) - gain(s, FINAL, boot))
        rows.append(contrast_row(
            "P33b", "fold gain: zero-shot - final", "+".join(sets),
            float(np.mean(pts)), np.mean(boots, axis=0)))
    for s in SETS:
        rows.append(contrast_row(
            "P33c", "final - zero-shot, family", s,
            point[(s, FINAL, "family")] - point[(s, ZERO, "family")],
            boot[(s, FINAL, "family")] - boot[(s, ZERO, "family")]))
    return rows


def evaluate(rng):
    rows, point, boot = [], {}, {}
    skipped = 0
    for s in SETS:
        refs = references(s)
        ids = list(refs)
        terms = pd.read_csv(TERMS / f"terms_{s}.csv", encoding="utf-8-sig",
                            dtype={"term": str}, usecols=["term"])["term"].tolist()
        rel = {"exact": relevance(refs, ids, terms, "exact"),
               "family": relevance(refs, ids, terms, "family")}
        for name, rel_key, _cond in SETTINGS:
            n_keep = int(rel[rel_key].any(axis=1).sum())
            n_skip = len(terms) - n_keep
            skipped += n_skip
            if n_skip:
                print(f"{s} {name}: skipped {n_skip} terms with 0 relevant", flush=True)
        # One shared index matrix per set, so every system and setting is paired.
        idx = rng.integers(0, len(terms), size=(N_BOOT, len(terms)))
        for sys_name, pat in SYSTEMS.items():
            path = Path(pat.format(s))
            if not path.exists():
                continue
            hyp = {r["utt_id"]: norm(r["hyp"]) for r in read_preds(path)}
            missing = [u for u in ids if u not in hyp]
            if missing:
                raise SystemExit(f"{path}: {len(missing)} of {len(ids)} clips missing")
            for name, rel_key, cond in SETTINGS:
                tp, fp, fn, recall, precision = rates(detection(hyp, ids, terms, cond), rel[rel_key])
                n_terms, micro_r, micro_p = micro(tp, fp, fn)
                r = float(defined_mean(recall)[0])
                p = float(defined_mean(precision)[0])
                f1 = float(f1_of(r, p))
                rows.append({"set": s, "system": sys_name, "setting": name, "terms": n_terms,
                             "macro_recall": 100 * r, "macro_precision": 100 * p, "f1": 100 * f1,
                             "micro_recall": 100 * micro_r, "micro_precision": 100 * micro_p})
                point[(s, sys_name, name)] = 100 * f1
                boot[(s, sys_name, name)] = 100 * f1_of(
                    defined_mean(recall[idx]), defined_mean(precision[idx]))
            print(f"{s} {sys_name}: done", flush=True)
    return pd.DataFrame(rows, columns=SUMMARY_COLS), contrasts_of(point, boot), skipped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [TERMS / f"terms_{s}.csv" for s in SETS]
    for s in SETS:
        inputs += [Path(p.format(s)) / "predictions.jsonl" for p in SYSTEMS.values() if Path(p.format(s)).exists()]
    with RunRecord(OUT / "run_info.json", inputs=inputs) as rec:
        summary, contrast_rows, skipped = evaluate(np.random.default_rng(SEED))
        contrasts = pd.DataFrame(contrast_rows, columns=CONTRAST_COLS)
        summary_path, contrast_path = OUT / "summary.csv", OUT / "contrasts.csv"
        summary.to_csv(summary_path, index=False)
        contrasts.to_csv(contrast_path, index=False)
        rec.outputs = [summary_path, contrast_path]
        print(f"terms skipped (0 relevant): {skipped}")
        print(summary.round(2).to_string(index=False))
        print(contrasts.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
