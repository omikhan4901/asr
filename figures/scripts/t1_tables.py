"""
Tables for the README (Bangla text: matplotlib cannot shape Bengali script, so these are Markdown):
  T-confusions  the 20 most frequent substitution pairs of zero-shot tugstugi on DL Sprint eval, with the
                rule-based type and how often the same pair is still made by M1 full and by Phase 3.
  T-examples    real DL Sprint eval clips: reference, zero-shot, Phase 3 (no training) and M1 full output,
                words that differ from the reference in bold. Chosen by a fixed rule (seed 0): 6-14
                reference words, zero-shot WER >= 25%, M1 full WER 0.

Inputs: results/error_types/{splits_fa-eval/<tugstugi>,phase3/splits_fa-eval/mbr_lm_x,phase4/splits_fa-eval/m1_full}/errors.csv,
        results/modal/{zeroshot,finetuned}/splits_fa-eval/<run>/predictions.jsonl,
        results/phase3/splits_fa-eval/mbr_lm_x/predictions.jsonl
Output: figures/data/t_confusions.csv, figures/data/t_examples.csv, figures/tables/*.md
Run:    PYTHONPATH=src ./Scripts/python.exe figures/scripts/t1_tables.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bangla_asr.common.stats import edit_counts  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

TUG = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
ERR = {"zero-shot": ROOT / "results/error_types/splits_fa-eval" / TUG / "errors.csv",
       "List correction": ROOT / "results/error_types/phase3/splits_fa-eval/mbr_lm_x/errors.csv",
       "M1 full": ROOT / "results/error_types/phase4/splits_fa-eval/m1_full/errors.csv"}
TYPE = {"convention": "convention", "vowel_shift": "vowel shift", "segmentation": "segmentation",
        "near_miss": "near miss", "substitution": "substitution", "number": "number"}
(ROOT / "figures/tables").mkdir(exist_ok=True)

# Confusion pairs.
counts = {}
for name, f in ERR.items():
    e = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
    e = e[e.op == "substitute"]
    counts[name] = e.groupby(["ref_word", "hyp_word"]).size()
    if name == "zero-shot":
        types = e.groupby(["ref_word", "hyp_word"])["type"].agg(lambda s: s.mode().iat[0])
top = counts["zero-shot"].sort_values(ascending=False).head(20)
rows = [{"reference": r, "model wrote": h, "type": TYPE.get(types[(r, h)], types[(r, h)]),
         "zero-shot": int(n), "List correction": int(counts["List correction"].get((r, h), 0)),
         "M1 full": int(counts["M1 full"].get((r, h), 0))} for (r, h), n in top.items()]
conf = pd.DataFrame(rows)
conf.to_csv(ROOT / "figures/data/t_confusions.csv", index=False, encoding="utf-8")
md = ["| reference → model wrote | type | zero-shot | List correction | M1 full |", "|---|---|---:|---:|---:|"]
md += [f"| {r['reference']} → {r['model wrote']} | {r['type']} | {r['zero-shot']} | {r['List correction']} | {r['M1 full']} |"
       for r in rows]
(ROOT / "figures/tables/t_confusions.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# Qualitative examples.
def load(p):
    return {json.loads(l)["utt_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


zs = load(ROOT / "results/modal/zeroshot/splits_fa-eval/tugstugi_a100/predictions.jsonl")
p3 = load(ROOT / "results/phase3/splits_fa-eval/mbr_lm_x/predictions.jsonl")
ft = load(ROOT / "results/modal/finetuned/splits_fa-eval/m1_full/predictions.jsonl")


def wer(h, r):
    e = edit_counts(h, r)
    return e["word_edits"] / max(1, e["ref_words"])


def bold_diff(hyp, ref):
    """Normalized hypothesis with every word that is not matched to the reference in bold."""
    h, r = normalize_bangla_text(hyp).split(), normalize_bangla_text(ref).split()
    keep = set(range(len(h)))
    for op in Levenshtein.editops(r, h):
        if op.tag in ("replace", "insert"):
            keep.discard(op.dest_pos)
    return " ".join(w if i in keep else f"**{w}**" for i, w in enumerate(h))


cands = []
for u, z in zs.items():
    ref = z["ref"]
    n = len(normalize_bangla_text(ref).split())
    if 6 <= n <= 14 and u in p3 and u in ft and wer(z["hyp"], ref) >= 0.25 and wer(ft[u]["hyp"], ref) == 0:
        cands.append(u)
random.Random(0).shuffle(cands)
ex = []
for u in sorted(cands[:5]):
    ref = zs[u]["ref"]
    ex.append({"utt_id": u, "reference": normalize_bangla_text(ref), "zero-shot": bold_diff(zs[u]["hyp"], ref),
               "List correction": bold_diff(p3[u]["hyp"], ref), "M1 full": bold_diff(ft[u]["hyp"], ref)})
pd.DataFrame(ex).to_csv(ROOT / "figures/data/t_examples.csv", index=False, encoding="utf-8")
md = ["| clip | system | transcript |", "|---|---|---|"]
for e in ex:
    for k in ("reference", "zero-shot", "List correction", "M1 full"):
        md.append(f"| {e['utt_id'] if k == 'reference' else ''} | {k} | {e[k]} |")
(ROOT / "figures/tables/t_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"{len(conf)} confusion pairs, {len(ex)} examples (from {len(cands)} candidate clips)")
