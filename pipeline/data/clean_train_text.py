#!/usr/bin/env python3
"""
Clean training references with deterministic rules (no model, no LLM).

1. Non-speech tags in brackets ([মিউজিক], [সশব্দ হাসি]) are removed from the text:
   dev and eval references have none, so a model trained on them would insert them.
2. A chunk is dropped if its text has any letter or combining mark (Unicode
   category L* / M*) outside the Bengali block U+0980-U+09FF (Latin, Arabic,
   Devanagari, ...), or is empty after step 1.
Punctuation, digits, the danda (U+0964/0965, in the Devanagari block) and
ZWJ/ZWNJ are kept; the scoring normalizer handles punctuation anyway.
Training data only: dev and eval references are never changed.

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/data/clean_train_text.py
  -> data/splits_fa/train_clean.csv, data/splits_fa/train_clean_report.json
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bangla_asr.common.provenance import RunRecord
from bangla_asr.common.safe_io import atomic_write_text


TAG = re.compile(r"\[[^\]]*\]")


def strip_tags(text: str) -> str:
    return " ".join(TAG.sub(" ", text).split())


def foreign_letters(text: str) -> list:
    return [c for c in text if unicodedata.category(c)[0] in "LM" and not "\u0980" <= c <= "\u09ff"]


def script_of(c: str) -> str:
    return unicodedata.name(c, "UNKNOWN").split()[0]  # LATIN, ARABIC, DEVANAGARI, TELUGU, ...


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/splits_fa/train.csv")
    p.add_argument("--output", default="data/splits_fa/train_clean.csv")
    args = p.parse_args()
    out, report_path = Path(args.output), Path(args.output).with_name(Path(args.output).stem + "_report.json")

    with RunRecord(out.with_suffix(".run_info.json"), inputs=[args.input]) as rec:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        text = df["text"].astype(str)
        tagged = text.str.contains(TAG)
        df["text"] = text.map(strip_tags)
        bad = df["text"].map(foreign_letters)
        drop = bad.map(bool) | (df["text"] == "")
        scripts = Counter(sc for letters in bad[drop] for sc in {script_of(c) for c in letters})
        kept = df[~drop]
        tmp = out.with_suffix(".tmp")
        kept.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(out)
        report = {
            "rule": "strip [bracketed] tags; drop if any Unicode letter/mark outside U+0980-U+09FF or empty",
            "chunks_with_tags_stripped": int(tagged.sum()),
            "input_chunks": len(df), "dropped_chunks": int(drop.sum()), "kept_chunks": len(kept),
            "input_hours": round(df["duration"].sum() / 3600, 3), "kept_hours": round(kept["duration"].sum() / 3600, 3),
            "dropped_chunks_by_script": dict(scripts.most_common()),
            "dropped_empty": int((df["text"] == "").sum()),
            "dropped_utt_ids": df.loc[drop, "utt_id"].tolist(),
        }
        atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2))
        rec.outputs = [out, report_path]
    print(f"{len(df)} chunks: tags stripped in {report['chunks_with_tags_stripped']}; dropped {report['dropped_chunks']} "
          f"({report['dropped_chunks_by_script']}, empty {report['dropped_empty']}), "
          f"kept {len(kept)} ({report['kept_hours']} h) -> {out}")


if __name__ == "__main__":
    main()
