#!/usr/bin/env python3
"""
M6: the M5 merged training list with harmonized spelling (RQ3).

The four training sources write the same word differently (R22: DL Sprint always কি, ছিল, তার,
কোন; FLEURS/Kathbath/CV mix কি/কী, prefer কোনো, ...). M6 keeps M5's clips and audio and rewrites
only the target texts to ONE convention, DL Sprint's, so the train x test matrix can show what
unifying spelling does to each test set scored against its own original references.

Rules come from the training texts only (no test or dev references):
  1. variant classes: words with the same spelling-convention key (bangla_asr.common.variants.fold)
     and >= 2 surface forms, each seen >= --min_form times over the four pooled training sources;
  2. target form of a class = the form DL Sprint train uses, if DL Sprint train has >= --min_class
     tokens of the class and one form takes >= --min_share of them (DL Sprint is very uniform);
     classes without a clear DL Sprint form are left alone, and so are the classes whose forms are
     different words (DISTINCT_WORDS, checked by hand);
  3. every word of every M5 row that is a non-target form of a kept class becomes the target form
     (punctuation around the word is kept). DL Sprint rows change only where they use their own
     minority form.

Outputs (--out_dir): m6-harmonized.csv (same columns and rows as M5, text rewritten),
rules.csv (class, forms, target, counts per source), report.json (words and rows changed per source).

Usage:
  PYTHONPATH=src ./Scripts/python.exe pipeline/data/harmonize_m6.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import fold  # noqa: E402

FV = "data/matrix/friend_volume"  # manifests pulled read-only from the Modal volume that trained M2-M5
SOURCES = {  # name -> training manifest (the full per-source list M5 sampled from)
    "dlsprint": "data/splits_fa/train_clean.csv",
    "cv": f"{FV}/cv-train.csv",
    "kathbath": f"{FV}/kathbath-train.csv",
    "shrutilipi": f"{FV}/shrutilipi-train.csv",
}
M5_SOURCE = {"train_clean": "dlsprint", "cv-train": "cv", "kathbath-train": "kathbath", "shrutilipi-train": "shrutilipi"}
# Classes the fold key merges although their forms are different words (checked by hand on rules.csv,
# 2026-10-11): আলো light / আল ridge, পুরো whole / পুর town, বার time / বারো twelve, এস / এসো, and
# single-letter forms. A class with any of these forms is never rewritten.
DISTINCT_WORDS = {"আল", "পুর", "বারো", "এস", "স", "সো", "ন", "নো"}


def words(text: str) -> list:
    return normalize_bangla_text(text).split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5", default=f"{FV}/m5-merged.csv")
    ap.add_argument("--out_dir", default="data/matrix/m6")
    ap.add_argument("--min_form", type=int, default=20, help="min count of a form, four sources pooled")
    ap.add_argument("--min_class", type=int, default=20, help="min class tokens in DL Sprint train")
    ap.add_argument("--min_share", type=float, default=0.9, help="min share of the DL Sprint form")
    args = ap.parse_args()

    counts = {}
    for name, path in SOURCES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        counts[name] = Counter(w for t in df["text"].astype(str) for w in words(t))
    pooled = sum(counts.values(), Counter())

    by_key = defaultdict(list)
    for w, n in pooled.items():
        if n >= args.min_form:
            by_key[fold(w)].append(w)
    classes = {k: sorted(v, key=lambda w: -pooled[w]) for k, v in by_key.items() if len(v) >= 2}

    rules, rows = {}, []
    for k, forms in sorted(classes.items(), key=lambda kv: -sum(pooled[w] for w in kv[1])):
        dl = Counter({w: counts["dlsprint"][w] for w in forms})
        n_dl = sum(dl.values())
        target, n_target = dl.most_common(1)[0]
        share = n_target / n_dl if n_dl else 0.0
        kept = n_dl >= args.min_class and share >= args.min_share and not DISTINCT_WORDS & set(forms)
        if kept:
            rules.update({w: target for w in forms if w != target})
        rows.append({"class": k, "forms": " / ".join(forms), "target": target if kept else "",
                     "kept": kept, "dlsprint_tokens": n_dl, "dlsprint_target_share": round(share, 3),
                     **{f"{s}|{w}": counts[s][w] for s in SOURCES for w in forms}})

    m5 = pd.read_csv(args.m5, encoding="utf-8-sig")
    changed_words, changed_rows, total_words = Counter(), Counter(), Counter()

    def rewrite(text: str, src: str) -> str:
        out, n = [], 0
        for tok in str(text).split():
            key = normalize_bangla_text(tok)  # the word without punctuation, NFC, canonical nukta
            if key in rules:
                n += 1
                # keep the punctuation around the word; if the raw token is not plain NFC, write the word alone
                tok = tok.replace(key, rules[key], 1) if key in tok else rules[key]
            out.append(tok)
        total_words[src] += len(out)
        changed_words[src] += n
        changed_rows[src] += n > 0
        return " ".join(out)

    m6 = m5.copy()
    m6["text"] = [rewrite(t, M5_SOURCE[s]) for t, s in zip(m5["text"], m5["source"])]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m6.to_csv(out / "m6-harmonized.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out / "rules.csv", index=False, encoding="utf-8-sig")
    n_rows = m5["source"].map(M5_SOURCE).value_counts()
    report = {
        "m5": args.m5, "sources": SOURCES, "min_form": args.min_form, "min_class": args.min_class,
        "min_share": args.min_share, "classes": len(classes), "classes_kept": sum(r["kept"] for r in rows),
        "rewrite_rules": len(rules),
        "per_source": {s: {"rows": int(n_rows.get(s, 0)), "rows_changed": int(changed_rows[s]),
                           "words": int(total_words[s]), "words_changed": int(changed_words[s]),
                           "words_changed_pct": round(100 * changed_words[s] / max(1, total_words[s]), 2)}
                       for s in SOURCES},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
