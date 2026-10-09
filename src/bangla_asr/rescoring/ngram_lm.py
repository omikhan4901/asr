"""
Word n-gram language model with interpolated Kneser-Ney smoothing (pure Python).

KenLM does not build on Windows, and our training text is small (DL Sprint
train transcripts, ~0.6M words), so a dictionary-based model is fast enough.

Definition (Chen & Goodman 1998, interpolated KN with one discount per order):
  P_k(w | h) = max(c_k(h w) - D_k, 0) / c_k(h .)
               + D_k * N1+(h .) / c_k(h .) * P_{k-1}(w | h[1:])
  - order n uses raw counts; lower orders use continuation counts
    c_k(g) = number of distinct words seen before g;
  - D_k = n1 / (n1 + 2 n2) from the count-of-counts of order k (0.5 when n1 or n2 is 0);
  - an unseen history falls through to the next lower order;
  - the unigram level interpolates with a uniform distribution over the
    vocabulary + 1, so unseen words get a small non-zero probability.
Sentences are padded with n-1 <s> and end with </s>, which is scored.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

BOS, EOS = "<s>", "</s>"


class KneserNeyLM:
    def __init__(self, sentences: Iterable[Sequence[str]], order: int = 3):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.n = order
        counts: Dict[int, Counter] = {order: Counter()}
        for s in sentences:
            toks = [BOS] * (order - 1) + list(s) + [EOS]
            for i in range(order - 1, len(toks)):
                counts[order][tuple(toks[i - order + 1 : i + 1])] += 1
        for k in range(order - 1, 0, -1):  # continuation counts from order k+1
            counts[k] = Counter(g[1:] for g in counts[k + 1])
        self.counts = counts
        self.ctx_sum: Dict[int, Dict[Tuple[str, ...], int]] = {}
        self.ctx_types: Dict[int, Dict[Tuple[str, ...], int]] = {}
        self.discount: Dict[int, float] = {}
        for k, ck in counts.items():
            tot, types = defaultdict(int), defaultdict(int)
            for g, c in ck.items():
                tot[g[:-1]] += c
                types[g[:-1]] += 1
            self.ctx_sum[k], self.ctx_types[k] = dict(tot), dict(types)
            coc = Counter(ck.values())
            n1, n2 = coc.get(1, 0), coc.get(2, 0)
            # D must lie in (0, 1); without singletons the estimate is 0 and unseen words would get P = 0.
            self.discount[k] = n1 / (n1 + 2 * n2) if n1 and n2 else 0.5
        self.vocab_size = len(counts[1])  # words + </s>; <s> is never predicted

    def prob(self, word: str, history: Tuple[str, ...]) -> float:
        """P(word | last n-1 words of history)."""
        h = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
        return self._p(len(h) + 1, h, word)

    def _p(self, k: int, h: Tuple[str, ...], w: str) -> float:
        tot = self.ctx_sum[k].get(h)
        if k == 1:
            lower = 1.0 / (self.vocab_size + 1)
        else:
            lower = self._p(k - 1, h[1:], w)
            if not tot:
                return lower
        d = self.discount[k]
        return max(self.counts[k].get(h + (w,), 0) - d, 0) / tot + d * self.ctx_types[k][h] / tot * lower

    def sentence_logprob(self, words: Sequence[str]) -> float:
        """Natural-log probability of the sentence, including </s>."""
        hist: List[str] = [BOS] * (self.n - 1)
        lp = 0.0
        for w in list(words) + [EOS]:
            lp += math.log(self.prob(w, tuple(hist)))
            hist.append(w)
        return lp

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Per-token perplexity (tokens include </s>)."""
        lp, n = 0.0, 0
        for s in sentences:
            lp += self.sentence_logprob(s)
            n += len(s) + 1
        return math.exp(-lp / max(1, n))
