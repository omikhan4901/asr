#!/usr/bin/env python3
"""
Step 10: GRPO fine-tuning of a small reasoning model with SWER-improvement reward.

Local:
    python pipeline/10_train_grpo.py --train_jsonl data/processed/correction/train.jsonl

Modal (same volume pattern as N-best generation):
    modal run src/bangla_asr/training/modal_train.py --train-jsonl /data/train_correction.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.config import GRPOConfig
from bangla_asr.training.train_grpo import run_training_from_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO correction training (JSON/CSV logs only).")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="WeiboAI/VibeThinker-3B")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_clip", type=float, default=2.0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--log_dir", type=str, default="logs/grpo")
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_id=args.model_id,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    result = run_training_from_jsonl(args.train_jsonl, args.eval_jsonl, config=cfg)
    print(result)


if __name__ == "__main__":
    main()
