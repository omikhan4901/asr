"""Acoustic search features shared by BERT training and inference."""

from typing import Dict, List

import numpy as np


def acoustic_features_for_pool(candidates: List[Dict]) -> List[List[float]]:
    """Returns [normalized_rank, z-scored logprob] for each candidate in a pool."""
    if not candidates:
        return []

    max_rank = max(c.get("rank", 0) for c in candidates) or 1
    logprobs = np.array(
        [float(c.get("logprob", 0.0)) for c in candidates],
        dtype=np.float64,
    )
    finite = np.isfinite(logprobs)
    z_scores = np.zeros_like(logprobs)
    if finite.any():
        mu = float(np.mean(logprobs[finite]))
        sigma = float(np.std(logprobs[finite]))
        if sigma >= 1e-6:
            z_scores = np.where(finite, (logprobs - mu) / sigma, 0.0)

    features = []
    for i, cand in enumerate(candidates):
        features.append([
            float(cand.get("rank", 0)) / float(max_rank),
            float(z_scores[i]),
        ])
    return features
