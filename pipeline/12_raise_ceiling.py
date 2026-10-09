#!/usr/bin/env python3
"""
Phase 3: raising the selection ceiling without retraining (PLAN E7, E8, E23, E24).

Candidate pool per utterance = the 12 sampled candidates + the greedy
hypothesis, each with its exact acoustic score (sum of token log-probabilities,
pipeline/rescore_nbest.py --add_greedy). Methods, all compared with greedy:

  am        rerank the pool by acoustic score + beta * words
  lm        + lambda * word n-gram LM log-probability (Kneser-Ney, DL Sprint train text)
  mbr_lm    MBR whose candidate posteriors come from the combined score (Novosad 2026:
            LM-posterior MBR helps where plain MBR does not); lambda = 0 is acoustic MBR
  rover     word-level combination of the pool on the greedy backbone (E24)
  rewrite   greedy rewritten by convention rules learned from greedy-vs-reference on TRAIN
  rewrite_sel  the same, only on greedy words that <= k sampled candidates agree with (E23)
  lm_x, mbr_lm_x  lm / mbr_lm on the EXPANDED pool: every candidate plus its rewrite
            (a rewrite inherits its parent's acoustic score; the LM decides the spelling)
Plus the oracle WER and the coverage miss rate (reference words no candidate has)
before and after expansion (E7), and the share of greedy's errors that are
selection-proof before and after.

Every weight, threshold and the LM order is chosen on the DL Sprint dev pool only
(splits_fa-dev) and applied unchanged to every set; test sets are never used for a
choice. The human labels are not used.

Outputs (--out_dir, default results/phase3/):
  <set>/<method>/predictions.jsonl  utt_id, group, ref, hyp (for error_types.py / paired_compare.py)
  summary.csv / summary.json        WER per set and method, paired difference vs greedy with a
                                    95% cluster-bootstrap CI (as paired_compare.py), oracle/coverage
  dev_grid.csv, tuned.json          every dev setting tried, and the chosen one per method
  rules.csv                         the chosen rewrite rules with their train counts
  run_info.json

Usage (after runs/phase3_ceiling.sh made the pools and the train greedy output):
  PYTHONPATH=src ./Scripts/python.exe pipeline/12_raise_ceiling.py
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.agreement import agreement_counts, correct_positions
from bangla_asr.common.provenance import RunRecord
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl
from bangla_asr.common.stats import bootstrap_ratio
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.config import MBRConfig
from bangla_asr.rescoring.combination import backbone_votes, combine
from bangla_asr.rescoring.convention import count_pairs, rewrite, select_rules
from bangla_asr.rescoring.mbr import softmax_with_temperature
from bangla_asr.rescoring.ngram_lm import KneserNeyLM

TUGP = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
TUNE = "splits_fa-dev"
POOLS = {
    TUNE: "data/phase3/dev_pool.jsonl",
    "splits_fa-eval": "data/phase3/eval_pool.jsonl",
    "fleurs-test": "data/phase3/fleurs_pool.jsonl",
    "kathbath-valid": "data/phase3/kathbath_pool.jsonl",
}
ORDERS = (2, 3, 4)
LAMS = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
BETAS = (-1, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8)
TEMPS = (0.5, 1, 2, 4, 8, 16)
MIN_COUNTS = (2, 3, 5, 10, 20)
MIN_PS = (0.3, 0.5, 0.7, 0.9)
KINDS = ("variant", "all")
EMPTY_LOSS = MBRConfig().empty_loss_penalty


def norm(text: str) -> List[str]:
    return normalize_bangla_text(text).split()


@dataclass
class Utt:
    utt_id: str
    group: str
    ref: List[str]
    toks: List[List[str]]          # candidate texts (normalized tokens)
    am: np.ndarray                 # acoustic score: sum of token log-probs
    g: int                         # index of the greedy candidate
    lm: Dict[int, np.ndarray] = field(default_factory=dict)
    nw: np.ndarray = None
    edits: np.ndarray = None
    loss: np.ndarray = None        # loss[i, j] = WER of candidate i against candidate j
    votes: list = None             # backbone votes of each candidate (greedy backbone)
    agree: np.ndarray = None       # per greedy word: how many sampled candidates have it


class LMs:
    def __init__(self, sentences, orders):
        self.models = {n: KneserNeyLM(sentences, n) for n in orders}
        self.cache = {n: {} for n in orders}

    def score(self, n, toks):
        key = " ".join(toks)
        c = self.cache[n]
        if key not in c:
            c[key] = self.models[n].sentence_logprob(toks)
        return c[key]


def pair_loss(toks):
    n = len(toks)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i, j] = EMPTY_LOSS if not toks[i] or not toks[j] else \
                    Levenshtein.distance(toks[j], toks[i]) / len(toks[j])
    return L


def features(u: Utt, lms: LMs):
    u.nw = np.array([len(t) for t in u.toks], float)
    u.edits = np.array([Levenshtein.distance(u.ref, t) for t in u.toks])
    u.lm = {n: np.array([lms.score(n, t) for t in u.toks]) for n in lms.models}
    u.loss = pair_loss(u.toks)


def load_pool(path, lms: LMs) -> List[Utt]:
    utts = []
    for r in tqdm(read_jsonl(path), desc=f"pool {Path(path).name}", mininterval=10):
        ref = norm(r["ground_truth"])
        if not ref:
            continue
        cands = r["candidates"]
        gi = [k for k, c in enumerate(cands) if c.get("source") == "greedy"]
        if len(gi) != 1:
            raise SystemExit(f"{path}: {r['utt_id']} has {len(gi)} greedy candidates (run rescore_nbest.py --add_greedy)")
        u = Utt(r["utt_id"], str(r["group"]), ref, [norm(c["text"]) for c in cands],
                np.array([c["sum_logprob"] for c in cands], float), gi[0])
        features(u, lms)
        back = u.toks[u.g]
        u.votes = [backbone_votes(back, t) for t in u.toks]
        u.agree = agreement_counts(back, [t for k, t in enumerate(u.toks) if k != u.g])
        utts.append(u)
    return utts


def expand(u: Utt, rules, lms: LMs) -> Utt:
    """Pool + the rewrite of every candidate that is not already in the pool (inherits its parent's score)."""
    toks, am, seen = list(u.toks), list(u.am), {" ".join(t) for t in u.toks}
    for t, a in zip(u.toks, u.am):
        r = rewrite(t, rules)
        if " ".join(r) not in seen:
            seen.add(" ".join(r))
            toks.append(r)
            am.append(a)
    x = Utt(u.utt_id, u.group, u.ref, toks, np.array(am), u.g)
    features(x, lms)
    return x


# ---- methods: (utt, params) -> (tokens, edits) ----

def combined(u, p):
    return u.am + p["lam"] * u.lm[p["order"]] + p["beta"] * u.nw


def m_rerank(u, p):
    i = int(np.argmax(combined(u, p)))
    return u.toks[i], int(u.edits[i])


def m_mbr(u, p):
    w = softmax_with_temperature(combined(u, p), p["T"])
    i = int(np.argmin(u.loss @ w))
    return u.toks[i], int(u.edits[i])


def m_rover(u, p):
    w = softmax_with_temperature(combined(u, p), p["T"])
    out = combine(len(u.toks[u.g]), u.votes, w).split()
    return out, Levenshtein.distance(u.ref, out)


def m_greedy(u, p):
    return u.toks[u.g], int(u.edits[u.g])


def make_rewrite(rule_sets):
    def m(u, p):
        out = rewrite(u.toks[u.g], rule_sets[p["rules"]], u.agree, p.get("max_agree"))
        return out, Levenshtein.distance(u.ref, out)
    return m


def corpus_wer(fn, utts, p):
    e = sum(fn(u, p)[1] for u in utts)
    return e / sum(len(u.ref) for u in utts)


def tune(name, fn, utts, grid, rows):
    best = None
    for p in tqdm(grid, desc=f"tune {name}", mininterval=10):
        w = corpus_wer(fn, utts, p)
        rows.append({"method": name, **{k: v for k, v in p.items()}, "dev_wer": w})
        if best is None or w < best[1] - 1e-12:
            best = (p, w)
    print(f"  {name:12s} dev WER {100 * best[1]:.2f}%  {best[0]}")
    for k, v in best[0].items():  # a choice at the end of its axis may want a wider grid
        axis = sorted({q[k] for q in grid if isinstance(q[k], (int, float))})
        if len(axis) > 1 and v in (axis[0], axis[-1]) and k != "max_agree" and not (k == "lam" and v == 0):
            print(f"  note: {name} {k}={v} is at the edge of its grid {axis[0]}..{axis[-1]}")
    return best[0]


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


def coverage(utts: List[Utt]):
    """Coverage miss rate and selection-proof share of greedy's reference-position errors."""
    miss = sp = gerr = words = 0
    for u in utts:
        covered = set()
        for t in u.toks:
            covered |= correct_positions(u.ref, t)[0]
        ok_g = correct_positions(u.ref, u.toks[u.g])[0]
        words += len(u.ref)
        miss += len(u.ref) - len(covered)
        err = [i for i in range(len(u.ref)) if i not in ok_g]
        gerr += len(err)
        sp += sum(i not in covered for i in err)
    return {"coverage_miss": miss / words, "greedy_errors_selection_proof": sp / max(1, gerr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="*", default=[f"{k}={v}" for k, v in POOLS.items()],
                    help="set=pool.jsonl ...; the first one must be the tuning set " + TUNE)
    ap.add_argument("--train_text", default="data/splits_fa/train_clean.csv")
    ap.add_argument("--train_greedy", default=f"results/zeroshot/splits_fa-train/{TUGP}/predictions.jsonl")
    ap.add_argument("--out_dir", default="results/phase3")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    pools = dict(x.split("=", 1) for x in args.pools)
    if next(iter(pools)) != TUNE:
        raise SystemExit(f"the first pool must be the tuning set {TUNE}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with RunRecord(out / "run_info.json", inputs=[args.train_text, args.train_greedy, *pools.values()]) as rec:
        train = pd.read_csv(args.train_text, encoding="utf-8-sig")
        lm_text = [t for t in (norm(x) for x in train["text"].astype(str)) if t]
        print(f"LM text: {len(lm_text)} sentences, {sum(map(len, lm_text))} words")
        lms = LMs(lm_text, ORDERS)

        data = {name: load_pool(path, lms) for name, path in pools.items()}
        dev = data[TUNE]

        # Rewrite rules: learned from tugstugi greedy vs reference on TRAIN.
        tg = read_jsonl(args.train_greedy)
        train_ids = set(train["utt_id"])
        if len({r["utt_id"] for r in tg} & train_ids) < len(train_ids):
            raise SystemExit(f"{args.train_greedy} covers {len(tg)} of {len(train_ids)} train utterances")
        stats = count_pairs((norm(r["ref"]), norm(r["hyp"])) for r in tg if r["utt_id"] in train_ids)
        rule_sets = {f"{k}|{c}|{q}": select_rules(stats, c, q, k)
                     for k, c, q in itertools.product(KINDS, MIN_COUNTS, MIN_PS)}

        # ---- tuning on dev only ----
        rows = []
        tuned = {"greedy": {}}
        tuned["am"] = tune("am", m_rerank, dev, grid(order=[ORDERS[0]], lam=[0], beta=BETAS), rows)
        tuned["lm"] = tune("lm", m_rerank, dev, grid(order=ORDERS, lam=LAMS, beta=BETAS), rows)
        order = tuned["lm"]["order"]
        tuned["mbr_lm"] = tune("mbr_lm", m_mbr, dev, grid(order=[order], lam=LAMS, beta=BETAS, T=TEMPS), rows)
        tuned["rover"] = tune("rover", m_rover, dev, grid(order=[order], lam=LAMS, beta=BETAS, T=TEMPS), rows)
        m_rw = make_rewrite(rule_sets)
        tuned["rewrite"] = tune("rewrite", m_rw, dev, grid(rules=list(rule_sets)), rows)
        best_rules = rule_sets[tuned["rewrite"]["rules"]]
        n_sampled = max(len(u.toks) - 1 for u in dev)
        tuned["rewrite_sel"] = tune("rewrite_sel", m_rw, dev,
                                    grid(rules=[tuned["rewrite"]["rules"]], max_agree=list(range(n_sampled + 1))), rows)
        xdata = {name: [expand(u, best_rules, lms) for u in tqdm(us, desc=f"expand {name}", mininterval=10)]
                 for name, us in data.items()}
        tuned["lm_x"] = tune("lm_x", m_rerank, xdata[TUNE], grid(order=[order], lam=LAMS, beta=BETAS), rows)
        tuned["mbr_lm_x"] = tune("mbr_lm_x", m_mbr, xdata[TUNE], grid(order=[order], lam=LAMS, beta=BETAS, T=TEMPS), rows)

        methods = {"greedy": (m_greedy, False), "am": (m_rerank, False), "lm": (m_rerank, False),
                   "mbr_lm": (m_mbr, False), "rover": (m_rover, False), "rewrite": (m_rw, False),
                   "rewrite_sel": (m_rw, False), "lm_x": (m_rerank, True), "mbr_lm_x": (m_mbr, True)}

        # ---- apply the dev choices to every set ----
        summary = []
        outputs = []
        for name, us in data.items():
            words = np.array([len(u.ref) for u in us])
            groups = [u.group for u in us]
            g_edits = np.array([u.edits[u.g] for u in us])
            row = {"set": name, "n_utterances": len(us), "n_groups": len(set(groups)), "ref_words": int(words.sum())}
            for m, (fn, on_x) in methods.items():
                src = xdata[name] if on_x else us
                res = [fn(u, tuned[m]) for u in src]
                e = np.array([x[1] for x in res])
                row[f"{m}_wer"] = e.sum() / words.sum()
                if m != "greedy":
                    d = bootstrap_ratio(groups, e - g_edits, words, n_boot=args.n_boot, seed=0)
                    row[f"{m}_diff"] = d["value"]
                    row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = d["ci95"]
                    row[f"{m}_p_better"] = float((d["draws"] < 0).mean())
                pdir = out / name / m
                pdir.mkdir(parents=True, exist_ok=True)
                pred = pdir / "predictions.jsonl"
                atomic_write_text(pred, "".join(json.dumps({"utt_id": u.utt_id, "group": u.group, "ref": " ".join(u.ref),
                                                            "hyp": " ".join(t)}, ensure_ascii=False) + "\n"
                                                for u, (t, _) in zip(src, res)))
                outputs.append(pred)
            for tag, src in (("pool", us), ("x", xdata[name])):
                row[f"oracle_{tag}_wer"] = sum(u.edits.min() for u in src) / words.sum()
                for k, v in coverage(src).items():
                    row[f"{k}_{tag}"] = v
            row["mean_pool_size"] = float(np.mean([len(u.toks) for u in us]))
            row["mean_pool_size_x"] = float(np.mean([len(u.toks) for u in xdata[name]]))
            row["lm_perplexity_refs"] = lms.models[order].perplexity(u.ref for u in us)
            summary.append(row)
            print(f"{name:15s} " + "  ".join(f"{m} {100 * row[m + '_wer']:.2f}" for m in methods)
                  + f"  | oracle {100 * row['oracle_pool_wer']:.2f} -> {100 * row['oracle_x_wer']:.2f}")

        dev_ppl = {n: lms.models[n].perplexity(u.ref for u in dev) for n in ORDERS}
        rules_df = pd.DataFrame([{"hyp": h, "ref": r, "count": stats.pair_count[(h, r)], "hyp_count": stats.hyp_count[h]}
                                 for h, r in best_rules.items()]).sort_values("count", ascending=False)
        rules_df.to_csv(out / "rules.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(rows).to_csv(out / "dev_grid.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, float_format="%.5f")
        meta = {"tuned_on": TUNE, "lm_order": order, "dev_perplexity_by_order": dev_ppl,
                "lm_text": args.train_text, "lm_sentences": len(lm_text), "n_rules": len(best_rules),
                "rule_source": args.train_greedy, "n_boot": args.n_boot,
                "normalizer": "bangla_asr.common.text.normalize_bangla_text"}
        atomic_write_text(out / "tuned.json", json.dumps({"meta": meta, "tuned": tuned}, ensure_ascii=False, indent=2))
        atomic_write_text(out / "summary.json", json.dumps({"meta": meta, "tuned": tuned, "sets": summary},
                                                           ensure_ascii=False, indent=2))
        rec.outputs = [out / f for f in ("summary.csv", "summary.json", "tuned.json", "dev_grid.csv", "rules.csv")] + outputs
        rec.extra = meta


if __name__ == "__main__":
    main()
