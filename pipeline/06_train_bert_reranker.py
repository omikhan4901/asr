#!/usr/bin/env python3
"""
Step 6: Train Discriminative BERT Candidate Reranker on Training Candidates.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.config import BERTRerankerConfig
from bangla_asr.dataset.manifest import load_nbest_jsonl
from bangla_asr.models.bert_reranker.trainer import train_bert_reranker


def main():
    parser = argparse.ArgumentParser(description="Step 6: Train BanglaBERT discriminative candidate reranker.")
    parser.add_argument("--train_nbest_jsonl", type=str, default="data/processed/nbest/train_nbest_candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bert_reranker")
    parser.add_argument("--model_name", type=str, default="csebuetnlp/banglabert")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    print(f"Loading training candidate pools from {args.train_nbest_jsonl}...")
    records = load_nbest_jsonl(args.train_nbest_jsonl)
    print(f"Loaded {len(records)} utterances for reranker training.")

    bert_cfg = BERTRerankerConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    ckpt_path = train_bert_reranker(
        train_records=records,
        output_dir=args.output_dir,
        config=bert_cfg,
    )
    print(f"Step 6 Complete. Trained BERT reranker checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
