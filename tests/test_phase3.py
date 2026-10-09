"""
Tests for the Phase 3 building blocks: n-gram LM, word-level combination, convention rewriter.
Run: PYTHONPATH=src ./Scripts/python.exe tests/test_phase3.py
"""

import io
import math
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from bangla_asr.rescoring.combination import backbone_votes, combine
from bangla_asr.rescoring.convention import count_pairs, rewrite, select_rules
from bangla_asr.rescoring.ngram_lm import EOS, KneserNeyLM

# 1. The LM is a probability distribution for every order and history
sents = [s.split() for s in ["a b c", "a b d", "b c a", "c a b c d", "d d a"]]
vocab = sorted({w for s in sents for w in s} | {EOS})
for n in (1, 2, 3, 4):
    lm = KneserNeyLM(sents, n)
    for h in [(), ("a",), ("a", "b"), ("zz", "a"), ("<s>", "<s>", "<s>")]:
        total = sum(lm.prob(w, h) for w in vocab) + lm.prob("UNSEEN", h)
        assert abs(total - 1) < 1e-9, (n, h, total)
lm = KneserNeyLM(sents, 3)
assert lm.prob("b", ("<s>", "a")) > lm.prob("c", ("<s>", "a"))  # seen continuation wins
assert lm.sentence_logprob("a b c".split()) > lm.sentence_logprob("c b a".split())
assert math.isfinite(lm.sentence_logprob(["never", "seen"]))
print("[PASS] n-gram LM.")

# 2. Combination: a word wins when the weighted majority has it, in any slot
back = "আমি ভাত খাই".split()
cands = [back, "আমি ভাত খাই না".split(), "তুমি ভাত খাই না".split(), "আমি খাই না".split()]
votes = [backbone_votes(back, c) for c in cands]
assert combine(len(back), votes, [1, 1, 1, 1]) == "আমি ভাত খাই না"      # insertion by 3 of 4
assert combine(len(back), votes, [5, 1, 1, 1]) == "আমি ভাত খাই"         # backbone outweighs
assert combine(len(back), votes, [0, 0, 1, 0]) == "তুমি ভাত খাই না"     # only one candidate counts
assert combine(len(back), votes, [0, 0, 0, 1]) == "আমি খাই না"          # deletion wins its slot
print("[PASS] word-level combination.")

# 3. Convention rewriter: rules from (ref, hyp) pairs, thresholds, variant filter, selective use
pairs = [("সে কি বলল".split(), "সে কী বললো".split())] * 3 + \
        [("সে কি করল".split(), "সে কী করল".split())] + \
        [("রাম এল".split(), "শ্যাম এল".split())] * 3
stats = count_pairs(pairs)
assert stats.pair_count[("কী", "কি")] == 4 and stats.hyp_count["কী"] == 4
r_all = select_rules(stats, min_count=3, min_p=0.5, kind="all")
assert r_all == {"কী": "কি", "বললো": "বলল", "শ্যাম": "রাম"}, r_all
r_var = select_rules(stats, min_count=3, min_p=0.5, kind="variant")
assert r_var == {"কী": "কি", "বললো": "বলল"}, r_var                       # name swap is not a variant
assert select_rules(stats, min_count=5, min_p=0.5) == {}
assert rewrite("সে কী বললো".split(), r_var) == "সে কি বলল".split()
assert rewrite("সে কী বললো".split(), r_var, agreement=[12, 12, 3], max_agree=6) == "সে কী বলল".split()
print("[PASS] convention rewriter.")
print("All Phase 3 tests passed.")
