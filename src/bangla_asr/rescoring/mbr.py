"""
Minimum Bayes Risk (MBR) consensus rescoring with temperature-scaled acoustic weighting.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from bangla_asr.common.metrics import compute_wer, compute_cer, compute_pairwise_loss
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.config import MBRConfig


def softmax_with_temperature(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes softmax with temperature scaling."""
    if x.size == 0:
        return x
    # A failed generation can leave NaN/inf scores in a JSONL record. Keep
    # those candidates selectable, but assign them negligible probability.
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, 1.0 / len(x), dtype=np.float64)
    floor = np.min(x[finite]) - 1_000.0
    x = np.where(finite, x, floor)
    scaled = x / max(1e-5, temperature)
    scaled = scaled - np.max(scaled)
    e = np.exp(scaled)
    return e / np.sum(e)


class MBRSelector:
    """
    Selects hypothesis minimizing expected word loss under acoustic posterior weights:
        h_MBR = argmin_i sum_j P(h_j | X) * L(h_i, h_j)
    """

    def __init__(self, config: MBRConfig = MBRConfig()):
        self.cfg = config

    def select(self, candidates: List[Dict]) -> Dict:
        """
        Receives candidates: [{'text': str, 'logprob': float, 'rank': int}, ...]
        Returns the chosen candidate dict with attached 'mbr_risk'.
        """
        if not candidates:
            return {"text": "", "logprob": 0.0, "rank": 0, "mbr_risk": 0.0}

        n = len(candidates)
        if n == 1:
            res = dict(candidates[0])
            res["mbr_risk"] = 0.0
            return res

        logprobs = np.array([c.get("logprob", 0.0) for c in candidates], dtype=np.float64)
        weights = softmax_with_temperature(logprobs, temperature=self.cfg.temperature)

        texts = [c.get("text", "") for c in candidates]
        risks = np.zeros(n, dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                loss = compute_pairwise_loss(texts[i], texts[j], empty_penalty=self.cfg.empty_loss_penalty)
                risks[i] += weights[j] * loss

        best_idx = int(np.argmin(risks))
        selected = dict(candidates[best_idx])
        selected["mbr_risk"] = float(risks[best_idx])
        return selected


def run_mbr_rescoring(
    nbest_records: List[Dict],
    output_dir: Optional[str] = None,
    config: MBRConfig = MBRConfig(),
) -> Dict:
    """
    Runs MBR rescoring over full N-best records.
    Computes WER and CER, and optionally saves metrics and per-utterance predictions.
    """
    selector = MBRSelector(config)
    hypotheses = []
    references = []
    rescored_records = []

    for rec in nbest_records:
        candidates = rec.get("candidates", [])
        if not candidates:
            continue
        chosen = selector.select(candidates)
        gt = rec.get("ground_truth", "")
        hypotheses.append(chosen["text"])
        references.append(gt)

        rescored_records.append({
            "utt_id": rec.get("utt_id", ""),
            "audio_path": rec.get("audio_path", ""),
            "ground_truth": gt,
            "selected_text": chosen["text"],
            "selected_rank": chosen.get("rank", 0),
            "selected_logprob": chosen.get("logprob", 0.0),
            "mbr_risk": chosen.get("mbr_risk", 0.0),
        })

    wer = compute_wer(hypotheses, references)
    cer = compute_cer(hypotheses, references)

    results = {
        "method": "MBR Consensus",
        "wer": wer,
        "cer": cer,
        "n_utterances": len(rescored_records),
        "temperature": config.temperature,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "mbr_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(out_path / "mbr_predictions.jsonl", "w", encoding="utf-8") as f:
            for item in rescored_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"MBR results and predictions saved to {out_path}")

    return results
