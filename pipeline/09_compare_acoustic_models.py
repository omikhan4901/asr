#!/usr/bin/env python3
"""
Step 9: Side-by-side Whisper vs Wav2Vec2 evaluation on held-out N-best pools.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.config import MBRConfig
from bangla_asr.dataset.manifest import load_nbest_jsonl
from bangla_asr.eval.compare_models import compare_acoustic_tracks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Whisper and Wav2Vec2 1-best / MBR / oracle (and SWER if available)."
    )
    parser.add_argument("--whisper_jsonl", type=str, required=True)
    parser.add_argument("--wav2vec2_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed/results/compare_models")
    parser.add_argument(
        "--whisper_mbr_temperature",
        type=float,
        default=0.5,
        help="Published Whisper MBR temperature. Do not change if reproducing 32.29%.",
    )
    parser.add_argument(
        "--wav2vec2_mbr_temperature",
        type=float,
        default=0.5,
        help="Separate CTC MBR temperature. Scores are not on the Whisper log-prob scale.",
    )
    args = parser.parse_args()

    tracks = {
        "whisper": load_nbest_jsonl(args.whisper_jsonl),
        "wav2vec2": load_nbest_jsonl(args.wav2vec2_jsonl),
    }
    mbr_configs = {
        "whisper": MBRConfig(temperature=args.whisper_mbr_temperature),
        "wav2vec2": MBRConfig(temperature=args.wav2vec2_mbr_temperature),
    }
    compare_acoustic_tracks(tracks, mbr_configs=mbr_configs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
