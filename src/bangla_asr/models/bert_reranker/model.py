"""
Discriminative BERT candidate reranker architecture for Bangla ASR.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

from bangla_asr.config import BERTRerankerConfig


class BanglaBERTReranker(nn.Module):
    """
    Discriminative neural reranker using a pretrained Bangla transformer encoder.
    Combines BERT pooled text representations with acoustic search features
    (e.g., beam rank and sequence logprobs) to predict candidate WER.
    """

    def __init__(self, config: BERTRerankerConfig = BERTRerankerConfig()):
        super().__init__()
        self.cfg = config

        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size

        in_features = hidden_size
        if config.use_acoustic_features:
            in_features += config.acoustic_feature_dim

        self.regressor = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(in_features, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        acoustic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting scalar candidate error (WER).
        Shape: (batch_size, 1)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pooler representation or CLS token embedding
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        if self.cfg.use_acoustic_features:
            if acoustic_features is None:
                acoustic_features = pooled.new_zeros(
                    (pooled.shape[0], self.cfg.acoustic_feature_dim)
                )
            expected_shape = (pooled.shape[0], self.cfg.acoustic_feature_dim)
            if acoustic_features.ndim != 2 or acoustic_features.shape != expected_shape:
                raise ValueError(
                    "acoustic_features must have shape "
                    f"(batch_size, {self.cfg.acoustic_feature_dim})"
                )
            features = torch.cat([pooled, acoustic_features.to(pooled.dtype)], dim=-1)
        else:
            features = pooled

        pred_error = self.regressor(features)
        return pred_error.squeeze(-1)
