"""E1b: error profile of the 499A eval 1-best (edit types, near-miss substitutions,
edge vs interior errors, WER by fuzzy-alignment-score bin, top confusion pairs).

Usage: PYTHONPATH=src ./Scripts/python.exe analysis/error_profile.py
"""
import json, sys, csv, collections, re
sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.common.metrics import align_tokens
from rapidfuzz.distance import Levenshtein

recs = [json.loads(l) for l in open("data/processed_naive_vad/nbest/eval_nbest_candidates.jsonl", encoding="utf-8")]
scores = {}
with open("data/processed_naive_vad/chunks/chunks_manifest_with_scores.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        scores[r["audio_path"].replace("\\", "/").split("/")[-1].replace(".wav", "")] = float(r["align_score"] or 0)

tot = collections.Counter()
bucket = collections.defaultdict(lambda: [0, 0])
pos = collections.Counter()
subs = collections.Counter()
per_utt = []
nc = collections.Counter()
for rec in recs:
    ref = normalize_bangla_text(rec["ground_truth"]).split()
    top = min(rec["candidates"], key=lambda c: c["rank"])
    hyp = normalize_bangla_text(top["text"]).split()
    nc[len(rec["candidates"])] += 1
    if not ref:
        continue
    ops = align_tokens(ref, hyp)
    errs = 0
    n = len(ref)
    for o in ops:
        if o.op == "equal":
            continue
        errs += 1
        tot[o.op] += 1
        ri = o.ref_index
        if ri is None:
            # insertion: locate by hyp position
            ri = int(o.hyp_index * n / max(1, len(hyp)))
        edge = "edge(first/last 3)" if (ri < 3 or ri >= n - 3) else "interior"
        pos[edge] += 1
        if o.op == "substitute":
            d = Levenshtein.distance(o.ref_token, o.hyp_token)
            tot["sub_chardist<=1"] += d <= 1
            tot["sub_chardist<=2"] += d <= 2
            subs[(o.ref_token, o.hyp_token)] += 1
    # space-agnostic: does joining all words fix it?
    tot["ref_words"] += n
    tot["errs"] += errs
    s = scores.get(rec["utt_id"], -1)
    b = "<60" if s < 60 else "60-75" if s < 75 else "75-90" if s < 90 else ">=90"
    bucket[b][0] += errs
    bucket[b][1] += n
    per_utt.append((errs / n, rec["utt_id"]))

print("n_utts", len(recs), "cands/utt", dict(nc))
print("corpus WER", tot["errs"] / tot["ref_words"])
print({k: tot[k] for k in ("substitute", "delete", "insert")})
print("subs with char-dist<=1: %.1f%%, <=2: %.1f%%" % (100 * tot["sub_chardist<=1"] / tot["substitute"], 100 * tot["sub_chardist<=2"] / tot["substitute"]))
print("edge vs interior errors:", dict(pos), "edge share of ref words ~", round(6 * len(recs) / tot["ref_words"], 3))
for b in ["<60", "60-75", "75-90", ">=90"]:
    e, n = bucket[b]
    if n:
        print(f"align score {b}: WER {e/n:.3f} on {n} words")
per_utt.sort()
import statistics
w = [x for x, _ in per_utt]
print("per-utt WER median", statistics.median(w), "p90", w[int(.9 * len(w))], "frac>0.6", sum(x > .6 for x in w) / len(w))
errs_sorted = sorted(w, reverse=True)
print("top substitutions:")
for (r, h), c in subs.most_common(30):
    print(c, r, "->", h)
