"""
Corpus WER/CER with cluster-bootstrap confidence intervals.

Utterances from one recording (or speaker) are correlated, so resampling
utterances independently understates uncertainty. We resample whole groups
with replacement and recompute corpus WER = sum(edits) / sum(reference words).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from rapidfuzz.distance import Levenshtein

from bangla_asr.common.text import normalize_bangla_text


def edit_counts(hyp: str, ref: str) -> Dict[str, int]:
    """Word and character edit counts of one utterance (normalized text)."""
    h, r = normalize_bangla_text(hyp), normalize_bangla_text(ref)
    hw, rw = h.split(), r.split()
    return {
        "word_edits": Levenshtein.distance(rw, hw),
        "ref_words": len(rw),
        "char_edits": Levenshtein.distance(r, h),
        "ref_chars": len(r),
    }


def corpus_rates(
    rows: Sequence[Dict],
    group_key: Optional[str] = "group",
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict:
    """rows: dicts with word_edits, ref_words, char_edits, ref_chars (+ group_key)."""
    rows = [r for r in rows if r["ref_words"] > 0]
    we = np.array([r["word_edits"] for r in rows], float)
    rw = np.array([r["ref_words"] for r in rows], float)
    ce = np.array([r["char_edits"] for r in rows], float)
    rc = np.array([r["ref_chars"] for r in rows], float)
    out = {
        "wer": float(we.sum() / rw.sum()) if len(rows) else float("nan"),
        "cer": float(ce.sum() / rc.sum()) if len(rows) else float("nan"),
        "n_utterances": len(rows),
        "ref_words": int(rw.sum()),
    }
    if not rows or n_boot <= 0:
        return out
    groups = [r.get(group_key, i) if group_key else i for i, r in enumerate(rows)]
    uniq = {g: k for k, g in enumerate(dict.fromkeys(groups))}
    gidx = np.array([uniq[g] for g in groups])
    G = len(uniq)
    gwe, grw = np.bincount(gidx, we, G), np.bincount(gidx, rw, G)
    gce, grc = np.bincount(gidx, ce, G), np.bincount(gidx, rc, G)
    rng = np.random.default_rng(seed)
    samp = rng.integers(0, G, size=(n_boot, G))
    wers = gwe[samp].sum(1) / np.maximum(1, grw[samp].sum(1))
    cers = gce[samp].sum(1) / np.maximum(1, grc[samp].sum(1))
    out.update({
        "n_groups": G,
        "wer_ci95": [float(np.percentile(wers, 2.5)), float(np.percentile(wers, 97.5))],
        "cer_ci95": [float(np.percentile(cers, 2.5)), float(np.percentile(cers, 97.5))],
    })
    return out
