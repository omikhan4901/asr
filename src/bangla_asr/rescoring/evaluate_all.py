"""
Unified benchmark evaluator comparing 1-Best Baseline, MBR Consensus,
BERT Discriminative Reranking, and the theoretical Oracle ceiling.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from bangla_asr.common.metrics import compute_wer, compute_cer
from bangla_asr.config import MBRConfig, BERTRerankerConfig
from bangla_asr.models.bert_reranker.ranker import BERTCandidateRanker
from bangla_asr.rescoring.mbr import MBRSelector
from bangla_asr.rescoring.oracle import evaluate_1best_baseline, evaluate_oracle_ceiling


class BenchmarkEvaluator:
    """Evaluates and compares all rescoring methods on an N-best evaluation set."""

    def __init__(
        self,
        mbr_config: MBRConfig = MBRConfig(),
        bert_checkpoint: Optional[str] = None,
        bert_config: Optional[BERTRerankerConfig] = None,
    ):
        self.mbr_selector = MBRSelector(mbr_config)
        self.bert_ranker = None
        if bert_checkpoint and Path(bert_checkpoint).exists():
            self.bert_ranker = BERTCandidateRanker(bert_checkpoint, bert_config)

    def evaluate(self, records: List[Dict], output_dir: Optional[str] = None) -> Dict:
        """Runs evaluation across all methods and calculates oracle gap recovery."""
        print(f"Running benchmark evaluation over {len(records)} utterances...")

        baseline = evaluate_1best_baseline(records)
        oracle = evaluate_oracle_ceiling(records)

        # MBR Consensus
        mbr_hyps, refs = [], []
        for rec in records:
            candidates = rec.get("candidates", [])
            if not candidates:
                continue
            chosen = self.mbr_selector.select(candidates)
            mbr_hyps.append(chosen["text"])
            refs.append(rec.get("ground_truth", ""))

        mbr_wer = compute_wer(mbr_hyps, refs)
        mbr_cer = compute_cer(mbr_hyps, refs)
        mbr_results = {
            "method": "MBR Consensus",
            "wer": mbr_wer,
            "cer": mbr_cer,
        }

        # BERT Discriminative Reranker (if loaded)
        bert_results = None
        if self.bert_ranker:
            bert_hyps = []
            for rec in records:
                candidates = rec.get("candidates", [])
                if not candidates:
                    continue
                chosen = self.bert_ranker.select_best_candidate(candidates)
                bert_hyps.append(chosen["text"])
            bert_wer = compute_wer(bert_hyps, refs)
            bert_cer = compute_cer(bert_hyps, refs)
            bert_results = {
                "method": "BERT Discriminative Reranker",
                "wer": bert_wer,
                "cer": bert_cer,
            }

        total_headroom = baseline["wer"] - oracle["wer"]
        mbr_recovered = ((baseline["wer"] - mbr_wer) / total_headroom * 100.0) if total_headroom > 0 else 0.0

        summary = {
            "n_utterances": len(records),
            "1best_baseline": baseline,
            "mbr_consensus": mbr_results,
            "oracle_ceiling": oracle,
            "oracle_headroom_points": round(total_headroom * 100.0, 2),
            "mbr_headroom_recovery_pct": round(mbr_recovered, 2),
        }

        if bert_results:
            bert_recovered = ((baseline["wer"] - bert_results["wer"]) / total_headroom * 100.0) if total_headroom > 0 else 0.0
            summary["bert_reranker"] = bert_results
            summary["bert_headroom_recovery_pct"] = round(bert_recovered, 2)

        # Print clean benchmark summary table
        print("\n" + "=" * 65)
        print("BANGLA ASR RESCORING BENCHMARK RESULTS")
        print("=" * 65)
        print(f"1-Best Baseline WER:     {baseline['wer']*100:.2f}% (CER: {baseline['cer']*100:.2f}%)")
        print(f"MBR Consensus WER:       {mbr_wer*100:.2f}% (CER: {mbr_cer*100:.2f}%)")
        print(f"  -> MBR Headroom Rec.:  {mbr_recovered:.1f}% of available oracle headroom")
        if bert_results:
            print(f"BERT Reranker WER:       {bert_results['wer']*100:.2f}% (CER: {bert_results['cer']*100:.2f}%)")
            print(f"  -> BERT Headroom Rec.: {bert_recovered:.1f}%")
        print(f"Oracle Ceiling WER:      {oracle['wer']*100:.2f}% (CER: {oracle['cer']*100:.2f}%)")
        print(f"Total Oracle Headroom:   {total_headroom*100:.2f} WER points")
        print("=" * 65)

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            with open(out_p / "benchmark_comparison.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"Saved benchmark results to {out_p / 'benchmark_comparison.json'}")

        return summary
