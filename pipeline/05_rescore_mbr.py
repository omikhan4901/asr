#!/usr/bin/env python3
"""
Step 5: Minimum Bayes Risk (MBR) Consensus Rescoring over N-Best Candidate Pools.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.config import MBRConfig
from bangla_asr.dataset.manifest import load_nbest_jsonl
from bangla_asr.rescoring.mbr import run_mbr_rescoring


def main():
    parser = argparse.ArgumentParser(description="Step 5: MBR consensus rescoring.")
    parser.add_argument("--nbest_jsonl", type=str, default="data/processed/nbest/eval_nbest_candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/processed/results/mbr")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for acoustic logprobs")
    args = parser.parse_args()

    print(f"Loading candidate records from {args.nbest_jsonl}...")
    records = load_nbest_jsonl(args.nbest_jsonl)
    print(f"Loaded {len(records)} utterances with candidate pools.")

    mbr_cfg = MBRConfig(temperature=args.temperature)
    results = run_mbr_rescoring(records, output_dir=args.output_dir, config=mbr_cfg)

    print("\n" + "=" * 50)
    print("MBR RESCORING SUMMARY")
    print("=" * 50)
    print(f"Utterances Evaluated: {results['n_utterances']}")
    print(f"MBR Word Error Rate (WER):      {results['wer']*100:.2f}%")
    print(f"MBR Character Error Rate (CER): {results['cer']*100:.2f}%")
    print(f"Temperature Scaling:            {results['temperature']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
