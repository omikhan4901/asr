"""E1c: learn hyp-word -> ref-word rewrite rules from TRAIN rank-0 alignments and
apply them to EVAL rank-0 (a lookup-table probe of systematic spelling/dialect errors).

Usage: PYTHONPATH=src ./Scripts/python.exe analysis/spelling_lookup.py
"""
import json, sys, collections
sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.common.metrics import align_tokens, compute_wer

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

train = load("data/processed_naive_vad/nbest/train_nbest_candidates.jsonl")
ev = load("data/processed_naive_vad/nbest/eval_nbest_candidates.jsonl")
top = lambda r: min(r["candidates"], key=lambda c: c["rank"])["text"]

hyp_count = collections.Counter()
pair = collections.Counter()
for r in train:
    ref = normalize_bangla_text(r["ground_truth"]).split()
    hyp = normalize_bangla_text(top(r)).split()
    if not ref:
        continue
    hyp_count.update(hyp)
    for o in align_tokens(ref, hyp):
        if o.op == "substitute":
            pair[(o.hyp_token, o.ref_token)] += 1

for min_c, min_p in [(5, 0.5), (10, 0.5), (5, 0.6), (3, 0.5)]:
    rules = {}
    for (h, rf), c in pair.items():
        if c >= min_c and c / hyp_count[h] > min_p:
            rules[h] = rf
    hyps = [" ".join(rules.get(w, w) for w in normalize_bangla_text(top(r)).split()) for r in ev]
    print(f"min_count={min_c} min_p={min_p}: {len(rules)} rules -> eval WER {compute_wer(hyps, [r['ground_truth'] for r in ev]):.4f}")
print("baseline eval WER", compute_wer([top(r) for r in ev], [r["ground_truth"] for r in ev]))
print("train utts", len(train))
