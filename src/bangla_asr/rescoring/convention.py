"""
Convention rewriter: word-level hyp -> ref rewrite rules learned from a model's
own errors on TRAIN (list correction, E7/E8/E23; dev-tuned successor of the R7 probe).

From (reference, hypothesis) pairs we count, for every hypothesis word h, how
often it is aligned to each reference word r (`variants.align_words`, which
pairs similar words when an alignment is ambiguous). A rule h -> r is kept when
the pair occurs >= min_count times and in >= min_p of h's occurrences.
kind="variant" keeps only rules whose pair is a spelling/dialect variant by the
rule-based typer (convention or vowel_shift), i.e. the convention transducer
proper; kind="all" also keeps other substitutions (names, frequent confusions).

Rules are context-free and one word to one word: merges and splits
(দু পাশে / দুপাশে) are not rewritten.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bangla_asr.common.variants import align_words, pair_type


@dataclass
class RuleStats:
    hyp_count: Counter    # occurrences of each hypothesis word
    pair_count: Counter   # (hyp word, ref word) substitution counts


def count_pairs(pairs: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> RuleStats:
    """pairs: (reference tokens, hypothesis tokens), already normalized."""
    hyp_count, pair_count = Counter(), Counter()
    for ref, hyp in pairs:
        hyp_count.update(hyp)
        for op in align_words(ref, hyp):
            if op.op == "substitute":
                pair_count[(op.hyp_token, op.ref_token)] += 1
    return RuleStats(hyp_count, pair_count)


def select_rules(stats: RuleStats, min_count: int, min_p: float, kind: str = "all") -> Dict[str, str]:
    if kind not in ("all", "variant"):
        raise ValueError(kind)
    best: Dict[str, Tuple[int, str]] = {}
    for (h, r), c in stats.pair_count.items():
        if c >= min_count and c / stats.hyp_count[h] >= min_p and c > best.get(h, (0, ""))[0]:
            if kind == "variant" and pair_type(r, h) not in ("convention", "vowel_shift"):
                continue
            best[h] = (c, r)
    return {h: r for h, (c, r) in best.items()}


def rewrite(tokens: Sequence[str], rules: Dict[str, str], agreement: Optional[Sequence[int]] = None,
            max_agree: Optional[int] = None) -> List[str]:
    """Apply the rules; with max_agree, only to words whose agreement count is <= max_agree (E23)."""
    if max_agree is None or agreement is None:
        return [rules.get(w, w) for w in tokens]
    return [rules.get(w, w) if a <= max_agree else w for w, a in zip(tokens, agreement)]
