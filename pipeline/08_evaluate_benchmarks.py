#!/usr/bin/env python3
"""
Step 8: Comprehensive Benchmark Evaluation (1-Best vs MBR vs BERT vs Oracle Ceiling).
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.config import MBRConfig, BERTRerankerConfig
from bangla_asr.dataset.manifest import load_nbest_jsonl
from bangla_asr.rescoring.evaluate_all import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="Step 8: Comparative benchmark evaluation.")
    parser.add_argument("--eval_nbest_jsonl", type=str, default="data/processed/nbest/eval_nbest_candidates.jsonl")
    parser.add_argument("--bert_checkpoint", type=str, default=None, help="Optional path to trained best_model.pt")
    parser.add_argument("--output_dir", type=str, default="data/processed/results/benchmark")
    parser.add_argument("--temperature", type=float, default=0.5, help="MBR softmax temperature")
    args = parser.parse_args()

    print(f"Loading candidate records from {args.eval_nbest_jsonl}...")
    records = load_nbest_jsonl(args.eval_nbest_jsonl)

    mbr_cfg = MBRConfig(temperature=args.temperature)
    evaluator = BenchmarkEvaluator(
        mbr_config=mbr_cfg,
        bert_checkpoint=args.bert_checkpoint,
    )
    evaluator.evaluate(records, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
