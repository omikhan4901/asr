"""
PyTorch Dataset for training and evaluating the BERT discriminative reranker on N-best candidates.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from bangla_asr.common.metrics import word_error_rate
from bangla_asr.config import BERTRerankerConfig
from bangla_asr.models.bert_reranker.features import acoustic_features_for_pool


class CandidatePairDataset(Dataset):
    """
    Constructs training instances from JSONL N-best candidate pools.
    Each item contains:
      - input_ids, attention_mask: tokenized hypothesis text
      - acoustic_features: [normalized_rank, length_normalized_logprob]
      - target_wer: scalar ground-truth WER of this candidate against reference
    """

    def __init__(
        self,
        nbest_records: List[Dict],
        tokenizer: AutoTokenizer,
        config: BERTRerankerConfig = BERTRerankerConfig(),
        is_training: bool = True,
    ):
        self.tokenizer = tokenizer
        self.cfg = config
        self.is_training = is_training
        self.samples = []

        self._build_samples(nbest_records)

    def _build_samples(self, records: List[Dict]):
        for rec in records:
            gt = rec.get("ground_truth", "")
            candidates = rec.get("candidates", [])
            if not candidates:
                continue

            pool_features = acoustic_features_for_pool(candidates)

            for cand, feats in zip(candidates, pool_features):
                hyp_text = cand.get("text", "")
                target_wer = 0.0
                if self.is_training:
                    if gt:
                        w = word_error_rate(hyp_text, gt)
                        target_wer = float(w) if not np.isnan(w) else 1.0
                    else:
                        target_wer = 1.0

                self.samples.append({
                    "text": hyp_text,
                    "norm_rank": feats[0],
                    "logprob": feats[1],
                    "target_wer": float(np.clip(target_wer, 0.0, 2.0)),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample["text"],
            max_length=self.cfg.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target_wer": torch.tensor(sample["target_wer"], dtype=torch.float32),
        }
        if self.cfg.use_acoustic_features:
            item["acoustic_features"] = torch.tensor(
                [sample["norm_rank"], sample["logprob"]], dtype=torch.float32
            )
        return item
