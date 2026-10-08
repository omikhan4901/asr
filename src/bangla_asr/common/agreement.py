"""
Candidate agreement: how many of the N candidates get each word right.

Used by the plateau anatomy (E3), error typing (E5) and the reference-free
agreement signal (E23). Alignment is ``align_tokens`` (the same one as in the
registry numbers R15/R16).
"""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from bangla_asr.common.metrics import align_tokens


def correct_positions(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[Set[int], Dict[int, Tuple[str, str]]]:
    """Reference indices the hypothesis gets right, and the op + hyp word per reference index."""
    ok, ops = set(), {}
    for o in align_tokens(ref, hyp):
        if o.ref_index is not None:
            ops[o.ref_index] = (o.op, o.hyp_token or "")
            if o.op == "equal":
                ok.add(o.ref_index)
    return ok, ops


def agreement_counts(target: Sequence[str], candidates: List[Sequence[str]]) -> np.ndarray:
    """c[i] = number of candidates that match word i of `target`.

    With target = reference this is the coverage count of E3 (c = 0 means
    selection-proof). With target = a hypothesis it is a reference-free
    confidence for each hypothesis word (E23).
    """
    counts = np.zeros(len(target), int)
    for cand in candidates:
        for i in correct_positions(target, cand)[0]:
            counts[i] += 1
    return counts
