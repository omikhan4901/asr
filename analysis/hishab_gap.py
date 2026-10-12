"""
Where does Hishab's FLEURS lead come from? Error decomposition of Hishab conformer_large vs our models on the
same utterances: substitutions / deletions / insertions, errors on number words and digits, Latin-script words,
spelling/dialect variants (same fold key as analysis/error_types.py), and how concentrated the gap is
(utterances sorted by edit difference).

Run: PYTHONPATH=src ./Scripts/python.exe analysis/hishab_gap.py [--set fleurs-test]
Out: results/hishab_gap/<set>.json (+ printed table)
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bangla_asr.common.metrics import align_tokens  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

DIGIT = re.compile(r"[0-9০-৯]")
LATIN = re.compile(r"[A-Za-z]")
NUMWORDS = set("এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ বিশ ত্রিশ চল্লিশ পঞ্চাশ ষাট সত্তর আশি নব্বই শত শো হাজার লক্ষ লাখ কোটি "
               "প্রথম দ্বিতীয় তৃতীয় চতুর্থ পঞ্চম শতাংশ".split())


def fold(w: str) -> str:
    """Crude spelling-variant key: drop vowel signs, hasanta, chandrabindu, nukta; unify য়/য, ড়/ড, ঢ়/ঢ, ী/ি, ূ/ু."""
    w = w.replace("য়", "য").replace("ড়", "ড").replace("ঢ়", "ঢ")
    return re.sub(r"[া-ৌৗ্ঁ়ৎ]", "", w)


def classify(r: str, h: str) -> str:
    if DIGIT.search(r) or DIGIT.search(h) or r in NUMWORDS or h in NUMWORDS:
        return "number"
    if LATIN.search(h) or LATIN.search(r):
        return "latin"
    if r and h and fold(r) == fold(h):
        return "variant"
    return "other"


def analyse(path: Path):
    rows = {json.loads(l)["utt_id"]: json.loads(l) for l in path.open(encoding="utf-8")}
    c, per_utt, nref = Counter(), {}, 0
    for u, r in rows.items():
        ref, hyp = normalize_bangla_text(r["ref"]).split(), normalize_bangla_text(r.get("hyp") or "").split()
        nref += len(ref)
        e = 0
        for op in align_tokens(ref, hyp):
            rt, ht = op.ref_token or "", op.hyp_token or ""
            if op.ref_index is None:
                c["ins"] += 1; c["ins_" + classify("", ht)] += 1; e += 1
            elif not ht:
                c["del"] += 1; c["del_" + classify(rt, "")] += 1; e += 1
            elif rt != ht:
                c["sub"] += 1; c["sub_" + classify(rt, ht)] += 1; e += 1
        per_utt[u] = e
    return {k: 100 * v / nref for k, v in c.items()}, per_utt, nref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fleurs-test")
    a = ap.parse_args()
    s = a.set
    systems = {
        "hishab": ROOT / f"results/zeroshot/{s}/hishab__titu_stt_bn_conformer_large/predictions.jsonl",
        "tugstugi zero-shot": ROOT / f"results/modal/zeroshot/{s}/tugstugi_a100/predictions.jsonl",
        "M1 full": ROOT / f"results/modal/finetuned/{s}/m1_full/predictions.jsonl",
        "M3 Kathbath": ROOT / f"results/kaggle/finetuned/{s}/m3_kathbath_t4/predictions.jsonl",
        "M5 merged": ROOT / f"results/kaggle/finetuned/{s}/m5_merged_t4/predictions.jsonl",
        "ROVER all": ROOT / f"results/ensemble/{s}/rover_all/predictions.jsonl",
    }
    res, utts = {}, {}
    for name, p in systems.items():
        if p.exists():
            res[name], utts[name], n = analyse(p)
    keys = ["sub", "del", "ins", "sub_variant", "sub_number", "del_number", "ins_number", "sub_latin", "sub_other",
            "del_other", "ins_other"]
    print(f"{s}: error rates in WER points (per 100 reference words), {n} reference words")
    print(f"{'system':20s} {'WER':>6s} " + " ".join(f"{k:>11s}" for k in keys))
    for name, r in res.items():
        print(f"{name:20s} {r['sub'] + r['del'] + r['ins']:6.2f} " + " ".join(f"{r.get(k, 0):11.2f}" for k in keys))
    out = {"set": s, "rates": res}
    for ours in ("M3 Kathbath", "ROVER all"):
        if ours in utts and "hishab" in utts:
            d = sorted(((utts[ours][u] - utts["hishab"][u], u) for u in utts["hishab"] if u in utts[ours]), reverse=True)
            gap = sum(x for x, _ in d)
            top = sum(x for x, _ in d[: len(d) // 10])
            better = sum(1 for x, _ in d if x < 0); worse = sum(1 for x, _ in d if x > 0)
            out[f"gap_{ours}"] = {"total_edit_gap": gap, "share_from_worst_10pct_utts": round(top / max(1, gap), 3),
                                  "utts_hishab_better": worse, "utts_ours_better": better, "worst_utts": d[:10]}
            print(f"{ours} vs hishab: edit gap {gap}, {100 * top / max(1, gap):.0f}% of it from the worst 10% of utterances; "
                  f"hishab better on {worse} utts, ours better on {better}")
    o = ROOT / "results/hishab_gap"; o.mkdir(parents=True, exist_ok=True)
    (o / f"{s}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
