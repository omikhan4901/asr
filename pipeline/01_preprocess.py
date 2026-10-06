#!/usr/bin/env python3
"""
Step 1: Long-Form Bangla Audio VAD Preprocessing and Sequential Ground-Truth Alignment.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_audio, save_audio
from bangla_asr.config import AudioConfig, VADConfig, AlignmentConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
from bangla_asr.preprocessing.vad import SileroVADChunker
from bangla_asr.preprocessing.aligner import SequentialFuzzyAligner


def main():
    parser = argparse.ArgumentParser(description="Step 1: Preprocess and align long-form Bangla audio.")
    parser.add_argument("--audio_dir", type=str, default="data/raw/train/audio", help="Path to raw WAV directory")
    parser.add_argument("--annotation_dir", type=str, default="data/raw/train/annotation", help="Path to reference text directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/chunks", help="Destination for chunk audio and manifest")
    parser.add_argument("--start_file", type=int, default=1, help="Start file index (e.g. 1 for train_001.wav)")
    parser.add_argument("--num_files", type=int, default=10, help="Number of files to process")
    parser.add_argument("--chunk_size", type=float, default=25.0, help="Max chunk length in seconds")
    parser.add_argument("--max_silence_gap_s", type=float, default=1.0, help="Do not merge speech spans across longer pauses")
    parser.add_argument("--split_search_window_s", type=float, default=1.0, help="Search this far backward for a quiet forced-split boundary")
    parser.add_argument("--min_alignment_score", type=int, default=50, help="RapidFuzz threshold")
    parser.add_argument("--rough_align_batch_size", type=int, default=8, help="Chunks per batched greedy-decode call during rough alignment (raise if VRAM allows, lower if it OOMs)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from checkpoint")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    audio_out_dir = out_path / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = out_path / "chunks_manifest_with_scores.csv"
    aligned_csv = out_path / "aligned_chunks.csv"
    checkpoint_file = out_path / "preprocess_checkpoint.json"

    completed_files = set()
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_files = set(json.load(f).get("completed_file_nums", []))

    audio_cfg = AudioConfig(max_chunk_duration_s=args.chunk_size)
    vad_cfg = VADConfig(
        max_silence_gap_s=args.max_silence_gap_s,
        split_search_window_s=args.split_search_window_s,
    )
    vad_chunker = SileroVADChunker(audio_cfg, vad_cfg)
    align_cfg = AlignmentConfig(min_alignment_score=args.min_alignment_score)
    aligner = SequentialFuzzyAligner(align_cfg)

    # Whisper model for rough alignment hypotheses
    whisper_cfg = WhisperConfig(num_beams=1, diverse_beam_search=False)
    whisper_wrapper = WhisperASRModel(whisper_cfg)

    file_range = list(range(args.start_file, args.start_file + args.num_files))
    remaining_files = [f for f in file_range if f not in completed_files]

    print(f"Starting Preprocessing on {len(remaining_files)} files (files {args.start_file} to {args.start_file + args.num_files - 1})...")

    for file_num in tqdm(remaining_files, desc="Processing Audio Files"):
        audio_name = f"train_{file_num:03d}.wav"
        text_name = f"train_{file_num:03d}.txt"

        audio_file = Path(args.audio_dir) / audio_name
        text_file = Path(args.annotation_dir) / text_name

        if not audio_file.exists() or not text_file.exists():
            continue

        try:
            audio_array, sr = load_audio(audio_file, target_sr=audio_cfg.sample_rate)
            raw_chunks = vad_chunker.segment_audio(audio_array)

            if not raw_chunks:
                completed_files.add(file_num)
                continue

            stem = audio_file.stem
            transcribed_chunks = []

            # Save every chunk's audio first (cheap I/O), then batch the rough
            # transcription pass through transcribe_batch() instead of one
            # generate() call per chunk -- greedy batch-size-1 decoding keeps
            # the GPU at "100% util" while launching tiny sequential kernels,
            # so throughput barely improves over CPU. Batching multiple
            # chunks per call is what actually uses the GPU's parallelism.
            chunk_meta = []
            for idx, c in enumerate(raw_chunks):
                chunk_filename = f"{stem}_{idx:03d}.wav"
                chunk_save_path = audio_out_dir / chunk_filename
                save_audio(chunk_save_path, c["audio"], sample_rate=sr)
                chunk_meta.append({
                    "chunk_id": idx,
                    "audio_path": str(chunk_save_path).replace("\\", "/"),
                    "audio": c["audio"],
                    "start": c["start_sec"],
                    "end": c["end_sec"],
                    "duration": c["duration_sec"],
                })

            for b_start in range(0, len(chunk_meta), args.rough_align_batch_size):
                batch = chunk_meta[b_start : b_start + args.rough_align_batch_size]
                hyp_texts = whisper_wrapper.transcribe_batch(
                    [m["audio"] for m in batch], sampling_rate=sr
                )
                for m, hyp_text in zip(batch, hyp_texts):
                    transcribed_chunks.append({
                        "chunk_id": m["chunk_id"],
                        "audio_path": m["audio_path"],
                        "text": hyp_text.strip(),
                        "start": m["start"],
                        "end": m["end"],
                        "duration": m["duration"],
                        "file_num": file_num,
                    })

            # Align with reference transcript
            ref_text = text_file.read_text(encoding="utf-8")
            aligned_chunks = aligner.align(transcribed_chunks, ref_text)

            # Persist incrementally
            file_df = pd.DataFrame(aligned_chunks)
            header = not manifest_csv.exists()
            file_df.to_csv(manifest_csv, mode="a", index=False, header=header, encoding="utf-8-sig")

            good_df = file_df[
                (file_df["ground_truth"].str.len() > 0) &
                (file_df["align_score"] >= args.min_alignment_score)
            ][["audio_path", "ground_truth", "align_score"]]
            good_df.rename(columns={"ground_truth": "gt"}, inplace=True)

            good_header = not aligned_csv.exists()
            good_df.to_csv(aligned_csv, mode="a", index=False, header=good_header, encoding="utf-8-sig")

            completed_files.add(file_num)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"completed_file_nums": sorted(completed_files)}, f)

        except Exception as exc:
            print(f"Error processing {audio_name}: {exc}")

    print("Step 1 Complete. Processed chunks written to:", out_path)


if __name__ == "__main__":
    main()
