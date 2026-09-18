"""
SWER-improvement reward for GRPO / any later RL algorithm.

The core signal is exactly:

    SWER(asr_hypothesis, ground_truth) - SWER(model_correction, ground_truth)

Scale and clip are config parameters, not hidden constants. This function
imports ``compute_swer`` from ``bangla_asr.metrics.swer`` so the train loop
cannot drift from the evaluation metric.

Empty references: SWER is NaN. By default those pairs are *excluded*
(``None``) rather than mapped to 0.0, because a zero reward still enters
the batch mean and dilutes the signal. Pass
``exclude_empty_reference=False`` only if you want the old zeroing behaviour.
"""

from __future__ import annotations

import math
from typing import Optional

from bangla_asr.config import SWERConfig
from bangla_asr.metrics.severity_lexicon import SeverityLexicon
from bangla_asr.metrics.swer import compute_swer, is_empty_reference


def swer_improvement_reward(
    asr_hypothesis: str,
    model_correction: str,
    ground_truth: str,
    *,
    scale: float = 1.0,
    clip: float = 2.0,
    lexicon: Optional[SeverityLexicon] = None,
    swer_config: Optional[SWERConfig] = None,
    exclude_empty_reference: bool = True,
) -> Optional[float]:
    """Return a clipped, scaled SWER delta, or ``None`` if the pair is excluded.

    Higher means the policy helped. ``None`` means "do not put this pair in
    the GRPO batch" (empty reference under the default flag).
    """
    if is_empty_reference(ground_truth):
        if exclude_empty_reference:
            return None
        return 0.0

    baseline = compute_swer(asr_hypothesis, ground_truth, lexicon=lexicon, config=swer_config)
    corrected = compute_swer(model_correction, ground_truth, lexicon=lexicon, config=swer_config)
    if math.isnan(baseline) or math.isnan(corrected):
        if exclude_empty_reference:
            return None
        return 0.0
    delta = float(baseline) - float(corrected)
    scaled = delta * float(scale)
    bound = abs(float(clip))
    return max(-bound, min(bound, scaled))
