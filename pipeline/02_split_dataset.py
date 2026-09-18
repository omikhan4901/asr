#!/usr/bin/env python3
"""
Step 2: Group-Aware Train/Eval Dataset Partitioning (Preventing Recording-Level Leakage).
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.dataset.manifest import load_aligned_manifest
from bangla_asr.dataset.splitter import group_aware_split


def main():
    parser = argparse.ArgumentParser(description="Step 2: Group-aware train/eval dataset split.")
    parser.add_argument("--aligned_csv", type=str, default="data/processed/chunks/aligned_chunks.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/splits")
    parser.add_argument("--eval_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading aligned chunks from {args.aligned_csv}...")
    df = load_aligned_manifest(args.aligned_csv)
    print(f"Total valid aligned utterances: {len(df)} from {df['source_file'].nunique()} unique source recordings.")

    train_df, eval_df = group_aware_split(df, eval_fraction=args.eval_frac, random_seed=args.seed)

    cols = ["utt_id", "audio_path", "gt", "source_file"]
    train_save = out_path / "train_split.csv"
    eval_save = out_path / "eval_split.csv"

    train_df[cols].to_csv(train_save, index=False, encoding="utf-8-sig")
    eval_df[cols].to_csv(eval_save, index=False, encoding="utf-8-sig")

    print("Step 2 Complete:")
    print(f"  Train: {len(train_df)} chunks ({train_df['source_file'].nunique()} sources) -> {train_save}")
    print(f"  Eval:  {len(eval_df)} chunks ({eval_df['source_file'].nunique()} sources) -> {eval_save}")


if __name__ == "__main__":
    main()
