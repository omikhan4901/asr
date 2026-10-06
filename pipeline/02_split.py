#!/usr/bin/env python3
"""
Step 2 (499B): recording-level train / dev / eval split of the forced-aligned chunks.

- eval = the recordings of the 499A eval split (so old and new numbers refer
  to the same recordings); dev = a few other recordings for tuning; train = rest.
- Quality filter uses only the aligner's own confidence (never an ASR model
  we evaluate). Unfiltered eval is also written (eval_all.csv) for the
  reference-noise / selection-bias analysis (E6).

Reads data/processed_fa/chunks/*.json, so it also works while 01_segment.py
is still running (only finished recordings are included).

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/02_split.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

COLS = ["utt_id", "recording", "audio_path", "start", "end", "duration", "n_words", "text",
        "mean_score", "min_score", "low_score_frac", "unaligned_words"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fa_dir", default="data/processed_fa")
    p.add_argument("--out_dir", default="data/splits_fa")
    p.add_argument("--old_eval_csv", default="data/processed_naive_vad/splits/eval_split.csv",
                   help="499A eval split; its recordings become the eval set")
    p.add_argument("--annotation_dir", default="data/raw/train/annotation",
                   help="All recordings with transcripts (dev is drawn from these, so it does not depend on progress)")
    p.add_argument("--n_dev", type=int, default=6, help="Recordings held out from train for tuning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_mean_score", type=float, default=0.5, help="Drop chunks whose mean word posterior is lower")
    p.add_argument("--max_low_frac", type=float, default=0.5, help="Drop chunks with more low-score words than this")
    p.add_argument("--min_dur", type=float, default=1.0)
    p.add_argument("--max_dur", type=float, default=25.5)
    args = p.parse_args()

    files = sorted(Path(args.fa_dir, "chunks").glob("*.json"))
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df["utt_id"] = df["recording"] + "_" + df.groupby("recording").cumcount().map("{:04d}".format)
    df = df[COLS]

    old_eval = pd.read_csv(args.old_eval_csv, encoding="utf-8-sig")
    eval_recs = sorted({u.rsplit("_", 1)[0] for u in old_eval["utt_id"]})
    all_recs = sorted(f.stem for f in Path(args.annotation_dir).glob("*.txt"))
    others = sorted(set(all_recs) - set(eval_recs))
    rng = np.random.default_rng(args.seed)
    dev_recs = sorted(rng.choice(others, size=min(args.n_dev, len(others)), replace=False).tolist()) if others else []

    df["split"] = np.where(df["recording"].isin(eval_recs), "eval",
                           np.where(df["recording"].isin(dev_recs), "dev", "train"))
    keep = (
        (df["mean_score"] >= args.min_mean_score)
        & (df["low_score_frac"] <= args.max_low_frac)
        & df["duration"].between(args.min_dur, args.max_dur)
        & (df["unaligned_words"] < df["n_words"])
    )
    df["kept"] = keep

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[df["split"] == "eval"].to_csv(out / "eval_all.csv", index=False, encoding="utf-8-sig")
    lines = [f"recordings segmented so far: {df['recording'].nunique()} "
             f"(eval {len(set(eval_recs) & set(df['recording']))}/{len(eval_recs)})",
             f"filter: mean_score >= {args.min_mean_score}, low_score_frac <= {args.max_low_frac}, "
             f"{args.min_dur}-{args.max_dur} s"]
    for split in ["train", "dev", "eval"]:
        part = df[df["split"] == split]
        kept = part[part["kept"]]
        kept.to_csv(out / f"{split}.csv", index=False, encoding="utf-8-sig")
        lines.append(f"{split:5s}: {kept['recording'].nunique():3d} recordings, {len(kept):6d} chunks, "
                     f"{kept['duration'].sum() / 3600:6.2f} h kept "
                     f"({100 * (1 - len(kept) / max(1, len(part))):.1f}% of chunks dropped)")
    (out / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "dev_recordings.txt").write_text("\n".join(dev_recs) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
