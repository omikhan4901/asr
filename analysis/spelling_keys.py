#!/usr/bin/env python3
"""
Spelling keys beyond a lookup table (PLAN §12, E25; pre-registered in PROGRESS §79).

Two words count as "the same word" for a detector when their keys are equal. Detectors:
  table      the learned word-to-word rewrite table of 12_raise_ceiling.py (results/list_correction/rules.csv)
  fold       bangla_asr.common.variants.fold (spelling-convention key, §3.5)
  deep       fold + the candidate rules kept by the screen below (FROZEN list KEPT)
  uzzaman04  UzZaman & Khan (2004), "A Bangla Phonetic Encoding for Better Spelling Suggestions",
             Table 3 + Figure 2 (prior work, a baseline; not our method)

  screen   for every candidate rule on top of fold: the DL Sprint train + dev error pairs (tugstugi greedy vs
           reference) that it newly merges, top 30 by count, for the written check; plus how many frequent
           training words it merges with another frequent word (homograph risk). No human label is read.
  score    the frozen detectors on the human-labelled word pairs (results/labels/pair_labels.csv):
           precision and recall of "spelling" vs "different word", seen vs unseen reference words, per stratum,
           pair-bootstrap CIs.

Outputs: results/spelling_keys/{screen_<rule>.csv, screen_summary.csv, summary.csv, pairs.csv, run_info.json}
  PYTHONPATH=src .venv/bin/python analysis/spelling_keys.py screen
  PYTHONPATH=src .venv/bin/python analysis/spelling_keys.py score
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import align_words, fold  # noqa: E402

TUG = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
GREEDY = [f"results/zeroshot/splits_fa-train/{TUG}/predictions.jsonl",
          f"results/zeroshot/splits_fa-dev/{TUG}/predictions.jsonl"]
TRAIN_TEXT = "data/backup/modal_friend/manifests/matrix/m5-300.csv"  # final model's training text (M1 data included)
TABLE = "results/list_correction/rules.csv"
PAIRS = "results/labels/pair_labels.csv"
OUT = Path("results/spelling_keys")
N_BOOT, SEED = 2000, 0

_V = "্"  # virama / hasanta
_CONS = "ক-হড়ঢ়য়"
_CONS_NO_RA = "ক-যল-হড়ঢ়য়"  # র্ব / র্য are reph + ব / য, not a phala

# Candidate rules (pre-registered list, PROGRESS §79), applied to the normalized word before fold().
# য (U+09AF) never stands for য় here: the normalizer writes য় as U+09DF.
CANDIDATES = {
    "sibilant": lambda w: re.sub("[শষ]", "স", w),                 # শ ষ -> স
    "nn": lambda w: w.replace("ণ", "ন"),                               # ণ -> ন
    "ja": lambda w: re.sub(f"(?<!{_V})য", "জ", w),                     # য -> জ (not য-ফলা)
    "rra": lambda w: re.sub("[ড়ঢ়]", "র", w),                      # ড় ঢ় -> র
    "vocalic_r": lambda w: w.replace("ঋ", "রি").replace("ৃ", _V + "রি"),  # ঋ -> রি, ৃ -> ্রি
    "ksha": lambda w: w.replace("ক" + _V + "ষ", "খ"),            # ক্ষ -> খ
    "ba_phala": lambda w: re.sub(f"(?<=[{_CONS_NO_RA}]){_V}ব", "", w),     # ্ব after a consonant dropped
    "ya_phala": lambda w: re.sub(f"(?<=[{_CONS_NO_RA}]){_V}য", "", w),     # ্য after a consonant dropped
}
# FROZEN 2026-11-03 after the written check (results/spelling_keys/screen_check.csv: kept when the top-30 train/dev
# error pairs a rule newly merges hold more same-word than different-word pairs; rra 12 vs 17 dropped). The reph
# exclusion of ba_phala/ya_phala was a definition fix made from the collision lists, before any label was read.
# Never changed after `score` ran.
KEPT = ["sibilant", "nn", "ja", "vocalic_r", "ksha", "ba_phala", "ya_phala"]


ORDER = ["vocalic_r", "ksha", "ba_phala", "ya_phala", "ja", "nn", "sibilant"]  # clusters before single letters
                                                                            # (sibilant first would turn ক্ষ into ক্স)


def deep(word: str) -> str:
    for name in ORDER:
        if name in KEPT:
            word = CANDIDATES[name](word)
    return fold(word)


# UzZaman & Khan 2004, Table 3. Not coded: hasant, sign O, candrabindu. Not in the table, our choice:
# sign vocalic R (ৃ) gets ঋ's code, khanda ta (ৎ) gets ত's code, other characters pass through.
# The table lists ঢ় with U+09A2 (ঢ, already under "D"): read as U+09DD.
_UZ = {"0": "্োঁ", "a": "আা", "i": "ইঈিী", "u": "উঊুূ",
       "e": "এেঐৈ", "o": "অওঔৌ", "k": "কখ", "g": "গঘ",
       "m": "মঙং", "c": "চছ", "j": "যজঝ", "T": "টঠ",
       "D": "ডঢ", "r": "ঋরড়ঢ়ৃ", "n": "নণ", "t": "তথৎ",
       "d": "দধ", "p": "পফ", "b": "বভ", "y": "য়", "l": "ল",
       "s": "শসষ", "h": "হঃ"}
_UZ_MAP = {ch: code for code, chars in _UZ.items() for ch in chars}


def uzzaman04(word: str) -> str:
    """Figure 2: replace every letter by its code, drop code 0, return the whole string (no truncation)."""
    return "".join(c for c in (_UZ_MAP.get(ch, ch) for ch in word) if c != "0")


def norm(text: str):
    return normalize_bangla_text(text).split()


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def error_pairs() -> Counter:
    """(ref word, hyp word) substitution counts of tugstugi greedy on DL Sprint train + dev."""
    pairs = Counter()
    for path in GREEDY:
        for r in read_jsonl(path):
            for op in align_words(norm(r["ref"]), norm(r["hyp"])):
                if op.op == "substitute":
                    pairs[(op.ref_token, op.hyp_token)] += 1
    return pairs


def train_vocab() -> Counter:
    df = pd.read_csv(TRAIN_TEXT, encoding="utf-8-sig", usecols=["text"])
    return Counter(w for t in df["text"].astype(str) for w in norm(t))


def screen(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with RunRecord(OUT / "run_info_screen.json", inputs=[*GREEDY, TRAIN_TEXT]) as rec:
        pairs = error_pairs()
        vocab = train_vocab()
        frequent = [w for w, c in vocab.items() if c >= args.min_freq]
        rows = []
        for name, rule in CANDIDATES.items():
            merged = [(r, h, c) for (r, h), c in pairs.items()
                      if fold(r) != fold(h) and fold(rule(r)) == fold(rule(h))]
            merged.sort(key=lambda x: -x[2])
            pd.DataFrame(merged[:args.top], columns=["ref", "hyp", "count"]).to_csv(
                OUT / f"screen_{name}.csv", index=False, encoding="utf-8-sig")
            # Homograph risk: frequent training words with different fold keys that the rule gives one key.
            groups = defaultdict(dict)  # rule key -> fold key -> most frequent word with that fold key
            for w in sorted(frequent, key=lambda x: -vocab[x]):
                groups[fold(rule(w))].setdefault(fold(w), w)
            coll = [list(g.values()) for g in groups.values() if len(g) > 1]
            coll.sort(key=lambda ws: -vocab[ws[1]])  # by the frequency of the second most frequent word
            pd.DataFrame([{"words": " / ".join(ws), "counts": " / ".join(str(vocab[w]) for w in ws)}
                          for ws in coll[:args.top]]).to_csv(OUT / f"screen_{name}_collisions.csv", index=False,
                                                             encoding="utf-8-sig")
            n_fold = len({fold(w) for w in frequent})
            rows.append({"rule": name, "merged_pairs": len(merged), "merged_tokens": sum(c for *_, c in merged),
                         "frequent_words": len(frequent), "fold_keys": n_fold, "keys_lost": n_fold - len(groups),
                         "collision_groups": len(coll)})
        summ = pd.DataFrame(rows)
        summ.to_csv(OUT / "screen_summary.csv", index=False)
        rec.outputs = [OUT / "screen_summary.csv", *(OUT / f"screen_{n}{s}.csv" for n in CANDIDATES
                                                     for s in ("", "_collisions"))]
        print(summ.to_string(index=False))


def pair_labels() -> pd.DataFrame:
    """One final label per labelled pair: the label with the largest total weight (adjudicated where needed)."""
    P = pd.read_csv(PAIRS, encoding="utf-8-sig")
    lab = P.groupby(["id", "label"])["w"].sum().reset_index().sort_values("w").groupby("id").tail(1)
    meta = P.drop_duplicates("id")[["id", "ref_word", "hyp_word", "rule_type", "stratum"]]
    out = meta.merge(lab[["id", "label"]], on="id")
    return out[out["label"] != "unsure"].reset_index(drop=True)


def score(args):
    if not KEPT and not args.allow_empty:
        raise SystemExit("KEPT is empty: run `screen`, do the written check, freeze KEPT first")
    OUT.mkdir(parents=True, exist_ok=True)
    with RunRecord(OUT / "run_info.json", inputs=[PAIRS, TABLE, TRAIN_TEXT, OUT / "screen_check.csv"]) as rec:
        T = pd.read_csv(TABLE, encoding="utf-8-sig")
        table = dict(zip(T["hyp"], T["ref"]))
        vocab = train_vocab()
        D = pair_labels()
        r, h = D["ref_word"].map(lambda x: normalize_bangla_text(str(x))), D["hyp_word"].map(lambda x: normalize_bangla_text(str(x)))
        D["seen"] = r.map(lambda w: w in vocab)
        D["table"] = [table.get(b) == a or table.get(a) == b for a, b in zip(r, h)]
        for name, key in (("fold", fold), ("deep", deep), ("uzzaman04", uzzaman04)):
            D[name] = [key(a) == key(b) for a, b in zip(r, h)]
        D.to_csv(OUT / "pairs.csv", index=False, encoding="utf-8-sig")

        rng = np.random.default_rng(SEED)
        rows = []
        for subset, S in [("all", D), *((f"stratum:{s}", g) for s, g in D.groupby("stratum")),
                          ("seen", D[D["seen"]]), ("unseen", D[~D["seen"]])]:
            pos, neg, dia = S["label"] == "spelling", S["label"] == "different", S["label"] == "dialect"
            idx = np.arange(len(S))
            boots = rng.integers(0, len(S), size=(N_BOOT, len(S))) if len(S) else None
            for det in ("table", "fold", "deep", "uzzaman04"):
                m = S[det].to_numpy()
                p, n, d = pos.to_numpy(), neg.to_numpy(), dia.to_numpy()

                def ratio(a, b):  # undefined (NaN) when nothing was merged / nothing to find
                    return a / b if b else np.nan

                def stats(i):
                    tp, fp = (m[i] & p[i]).sum(), (m[i] & n[i]).sum()
                    return (ratio(tp, (m[i] & (p[i] | n[i])).sum()), ratio(tp, p[i].sum()),
                            ratio((m[i] & d[i]).sum(), d[i].sum()), fp)
                prec, rec_, dmerge, fp = stats(idx)
                ci = np.array([stats(b)[:2] for b in boots]) if boots is not None else np.full((1, 2), np.nan)
                rows.append({"subset": subset, "detector": det, "spelling": int(p.sum()), "different": int(n.sum()),
                             "dialect": int(d.sum()), "precision": prec,
                             "precision_lo": np.nanpercentile(ci[:, 0], 2.5), "precision_hi": np.nanpercentile(ci[:, 0], 97.5),
                             "recall": rec_, "recall_lo": np.nanpercentile(ci[:, 1], 2.5),
                             "recall_hi": np.nanpercentile(ci[:, 1], 97.5), "false_merges": int(fp),
                             "dialect_merge_rate": dmerge})
        summ = pd.DataFrame(rows)
        summ.to_csv(OUT / "summary.csv", index=False)
        rec.outputs = [OUT / "summary.csv", OUT / "pairs.csv"]
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--min_freq", type=int, default=5)
    s.set_defaults(func=screen)
    c = sub.add_parser("score")
    c.add_argument("--allow_empty", action="store_true", help="score with no kept rule (deep == fold)")
    c.set_defaults(func=score)
    a = ap.parse_args()
    a.func(a)
