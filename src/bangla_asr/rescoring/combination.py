"""
Word-level combination of N-best candidates on a backbone (ROVER-style, E24).

Every candidate is aligned to one backbone hypothesis (we use greedy). Each
backbone word is a slot where every candidate votes for its aligned word, or
for "" when it deletes it; the gaps before, between and after backbone words
are insertion slots where a candidate votes for the words it inserts there
(joined into one phrase). Votes carry the candidate's posterior weight. In each
slot the heaviest option wins; in insertion slots "" also gets the weight of
every candidate that inserts nothing. Unlike MBR, the output can mix words from
different candidates, so it can reach words no single candidate gets right
everywhere (its bound is the compositional oracle, PLAN E3).
"""

from collections import defaultdict
from typing import List, Sequence, Tuple

from bangla_asr.common.metrics import align_tokens

Votes = List[Tuple[int, str]]  # (slot, option) for one candidate


def backbone_votes(backbone: Sequence[str], cand: Sequence[str]) -> Votes:
    """Slots: 2*i+1 = backbone word i; 2*i = insertion gap before word i (2*len = after the last)."""
    votes, ins, gap = [], defaultdict(list), 0
    for op in align_tokens(backbone, cand):
        if op.ref_index is not None:
            votes.append((2 * op.ref_index + 1, op.hyp_token or ""))
            gap = op.ref_index + 1
        else:
            ins[gap].append(op.hyp_token)
    votes.extend((2 * g, " ".join(ws)) for g, ws in ins.items())
    return votes


def combine(n_backbone: int, cand_votes: List[Votes], weights: Sequence[float]) -> str:
    slots = [defaultdict(float) for _ in range(2 * n_backbone + 1)]
    for votes, w in zip(cand_votes, weights):
        for s, opt in votes:
            slots[s][opt] += w
    total = float(sum(weights))
    out = []
    for s, opts in enumerate(slots):
        if s % 2 == 0:
            opts[""] += total - sum(opts.values())
        best = max(opts, key=opts.get) if opts else ""
        if best:
            out.append(best)
    return " ".join(out)
