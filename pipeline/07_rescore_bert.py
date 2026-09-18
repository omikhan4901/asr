#!/usr/bin/env python3
"""
Step 7: Evaluate Candidates using Trained Discriminative BanglaBERT Reranker.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.metrics import compute_wer, compute_cer
from bangla_asr.dataset.manifest import load_nbest_jsonl
from bangla_asr.models.bert_reranker.ranker import BERTCandidateRanker


def main():
    parser = argparse.ArgumentParser(description="Step 7: Rescore candidates with trained BERT reranker.")
    parser.add_argument("--eval_nbest_jsonl", type=str, default="data/processed/nbest/eval_nbest_candidates.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/bert_reranker")
    parser.add_argument("--output_dir", type=str, default="data/processed/results/bert")
    args = parser.parse_args()

    ckpt_file = Path(args.checkpoint_dir) / "best_model.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}. Train first with Step 6.")

    print(f"Loading evaluation candidates from {args.eval_nbest_jsonl}...")
    records = load_nbest_jsonl(args.eval_nbest_jsonl)
    print(f"Loaded {len(records)} utterances.")

    # The training configuration beside best_model.pt records the original
    # Hugging Face encoder name. Passing checkpoint_dir as model_name would
    # fail because it contains tokenizer files and a state dict, not an
    # AutoModel checkpoint.
    ranker = BERTCandidateRanker(str(ckpt_file))

    hypotheses = []
    references = []
    predictions = []

    print("Scoring candidates with BERT reranker...")
    for rec in records:
        gt = rec.get("ground_truth", "")
        candidates = rec.get("candidates", [])
        if not candidates:
            continue
        best_cand = ranker.select_best_candidate(candidates)

        hypotheses.append(best_cand["text"])
        references.append(gt)

        predictions.append({
            "utt_id": rec.get("utt_id", ""),
            "ground_truth": gt,
            "selected_text": best_cand["text"],
            "predicted_wer": best_cand.get("predicted_wer", 0.0),
            "original_rank": best_cand.get("rank", 0),
        })

    wer = compute_wer(hypotheses, references)
    cer = compute_cer(hypotheses, references)

    print("\n" + "=" * 55)
    print("BANGLABERT DISCRIMINATIVE RERANKER RESULTS")
    print("=" * 55)
    print(f"Evaluated Utterances:           {len(hypotheses)}")
    print(f"BERT-Rescored Word Error Rate (WER):      {wer*100:.2f}%")
    print(f"BERT-Rescored Character Error Rate (CER): {cer*100:.2f}%")
    print("=" * 55)

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "bert_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"method": "BERT Discriminative Reranker", "wer": wer, "cer": cer}, f, indent=2)
    with open(out_p / "bert_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved metrics and predictions to {out_p}")


if __name__ == "__main__":
    main()
