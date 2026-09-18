"""
Inference and candidate scoring with the trained BanglaBERTReranker.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer

from bangla_asr.config import BERTRerankerConfig
from bangla_asr.models.bert_reranker.features import acoustic_features_for_pool
from bangla_asr.models.bert_reranker.model import BanglaBERTReranker


class BERTCandidateRanker:
    """
    Ranks N-best candidate hypotheses using the trained discriminative BERT regressor.
    Selects the candidate that minimizes predicted WER.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[BERTRerankerConfig] = None,
        device: Optional[str] = None,
    ):
        checkpoint = Path(checkpoint_path)
        self.cfg = config or self._load_checkpoint_config(checkpoint)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Loading BERT Candidate Ranker from {checkpoint_path} on {self.device}...")
        tokenizer_source = checkpoint.parent if (checkpoint.parent / "tokenizer_config.json").exists() else self.cfg.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.model = BanglaBERTReranker(self.cfg).to(self.device)

        try:
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            saved = torch.load(checkpoint_path, map_location=self.device)
        state_dict = saved.get("model_state_dict", saved) if isinstance(saved, dict) else saved
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("BERT Ranker ready for scoring.")

    @staticmethod
    def _load_checkpoint_config(checkpoint_path: Path) -> BERTRerankerConfig:
        """Loads the training configuration saved beside a checkpoint.

        Older checkpoints did not have metadata, so they remain usable with
        the package defaults.
        """
        config_path = checkpoint_path.parent / "reranker_config.json"
        if not config_path.exists():
            return BERTRerankerConfig()
        with open(config_path, "r", encoding="utf-8") as f:
            values = json.load(f)
        allowed = {field.name for field in fields(BERTRerankerConfig)}
        return BERTRerankerConfig(**{key: value for key, value in values.items() if key in allowed})

    def score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Scores a list of candidate dicts [{'text': str, 'rank': int, 'logprob': float}, ...].
        Attaches 'predicted_wer' to each candidate dict.
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]

        tokens = self.tokenizer(
            texts,
            max_length=self.cfg.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        acoustic_tensors = None
        if self.cfg.use_acoustic_features:
            acoustic_tensors = torch.tensor(
                acoustic_features_for_pool(candidates),
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            preds = self.model(input_ids, attention_mask, acoustic_tensors)
            scores = preds.cpu().numpy().tolist()

        scored = []
        for cand, score in zip(candidates, scores):
            entry = dict(cand)
            entry["predicted_wer"] = float(score)
            scored.append(entry)

        return scored

    def select_best_candidate(self, candidates: List[Dict]) -> Dict:
        """Selects the candidate with the lowest predicted WER."""
        scored = self.score_candidates(candidates)
        if not scored:
            return {"text": "", "rank": 0, "logprob": 0.0, "predicted_wer": float("inf")}
        return min(scored, key=lambda c: c["predicted_wer"])
