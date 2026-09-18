#!/usr/bin/env python3
"""
Step 3: Autoregressive N-Best Candidate Generation using Whisper (Batched & Diverse Beam Search).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_audio
from bangla_asr.config import AudioConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
from bangla_asr.models.autoregressive.generator import BatchedNBestGenerator


def main():
    parser = argparse.ArgumentParser(description="Step 3: Generate N-best candidates with diverse beam search.")
    parser.add_argument("--split_csv", type=str, default="data/processed/splits/eval_split.csv")
    parser.add_argument("--output_jsonl", type=str, default="data/processed/nbest/eval_nbest_candidates.jsonl")
    parser.add_argument("--num_beams", type=int, default=12, help="Winning beam count from calibration sweep")
    parser.add_argument("--temperature", type=float, default=0.7, help="Stochastic beam sampling temperature")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset limit for fast testing")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_file.with_suffix(".checkpoint.json")

    completed_ids = set()
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_ids = set(json.load(f).get("completed_utt_ids", []))

    df = pd.read_csv(args.split_csv, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)

    remaining_df = df[~df["utt_id"].isin(completed_ids)].reset_index(drop=True)
    print(f"Loaded {len(df)} total utterances ({len(completed_ids)} already done, {len(remaining_df)} remaining).")

    whisper_cfg = WhisperConfig(
        num_beams=args.num_beams,
        diverse_beam_search=True,
        sampling_temperature=args.temperature,
        batch_size=args.batch_size,
    )
    model_wrapper = WhisperASRModel(whisper_cfg)
    generator = BatchedNBestGenerator(model_wrapper, whisper_config=whisper_cfg)

    batch_size = args.batch_size
    n_batches = (len(remaining_df) + batch_size - 1) // batch_size

    with open(out_file, "a", encoding="utf-8") as f_out:
        for b_idx in tqdm(range(n_batches), desc="Generating N-best Candidates"):
            batch_df = remaining_df.iloc[b_idx * batch_size : (b_idx + 1) * batch_size]
            audio_arrays = []
            valid_rows = []

            for _, row in batch_df.iterrows():
                try:
                    arr, _ = load_audio(row["audio_path"], target_sr=16000)
                    audio_arrays.append(arr)
                    valid_rows.append(row)
                except Exception as exc:
                    print(f"Failed to load audio for {row['utt_id']}: {exc}")

            if not audio_arrays:
                continue

            try:
                batch_candidates = generator.generate_batch(audio_arrays)
            except Exception as exc:
                print(f"Generation error on batch: {exc}")
                continue

            for row, candidates in zip(valid_rows, batch_candidates):
                record = {
                    "utt_id": row["utt_id"],
                    "audio_path": row["audio_path"],
                    "ground_truth": row["gt"],
                    "candidates": candidates,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed_ids.add(row["utt_id"])

            f_out.flush()
            with open(checkpoint_file, "w", encoding="utf-8") as f_chk:
                json.dump({"completed_utt_ids": sorted(completed_ids)}, f_chk)

    print(f"Step 3 Complete. Output candidate pools saved to {out_file}")


if __name__ == "__main__":
    main()
