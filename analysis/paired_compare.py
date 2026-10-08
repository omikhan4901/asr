"""
Paired comparison of two systems on the same utterances (E14; registry R17).

Each system is a predictions.jsonl from pipeline/eval_zero_shot.py, or an
N-best jsonl with a selector after a colon: <nbest>:rank0 | :mbr | :oracle.
WER difference B - A with a 95% cluster-bootstrap CI over recordings/speakers
(both systems resampled together), and P(B better) = share of resamples
where B has fewer edits.

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/paired_compare.py \
      --a results/zeroshot/splits_fa-eval/bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium/predictions.jsonl \
      --b data/nbest_fa/eval_nbest.jsonl:mbr --name eval_mbr_vs_greedy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.config import MBRConfig  # noqa: E402
from bangla_asr.rescoring.mbr import MBRSelector  # noqa: E402

SELECTORS = ("rank0", "mbr", "oracle")


def load_system(spec: str, mbr_temperature: float = 0.5):
    """{utt_id: (group, ref, hyp)} for a predictions file or an N-best file + selector."""
    path, sel = spec, None
    if ":" in spec and spec.rsplit(":", 1)[1] in SELECTORS:
        path, sel = spec.rsplit(":", 1)
    recs = read_jsonl(path)
    if sel is None:
        return path, {r["utt_id"]: (str(r["group"]), r["ref"], r["hyp"]) for r in recs}
    selector = MBRSelector(MBRConfig(temperature=mbr_temperature))
    out = {}
    for r in recs:
        cands = sorted(r["candidates"], key=lambda c: c["rank"])
        ref = r["ground_truth"]
        if sel == "rank0":
            hyp = cands[0]["text"]
        elif sel == "mbr":
            hyp = selector.select(cands)["text"]
        else:
            rw = normalize_bangla_text(ref).split()
            hyp = min(cands, key=lambda c: Levenshtein.distance(rw, normalize_bangla_text(c["text"]).split()))["text"]
        out[r["utt_id"]] = (str(r["group"]), ref, hyp)
    return path, out


def word_edits(ref: str, hyp: str):
    r, h = normalize_bangla_text(ref).split(), normalize_bangla_text(hyp).split()
    return Levenshtein.distance(r, h), len(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline system")
    p.add_argument("--b", required=True, help="compared system")
    p.add_argument("--name", required=True)
    p.add_argument("--out_dir", default="results/compare")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mbr_temperature", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out_dir, f"{args.name}.json")
    path_a, A = load_system(args.a, args.mbr_temperature)
    path_b, B = load_system(args.b, args.mbr_temperature)
    with RunRecord(out.with_suffix(".run_info.json"), inputs=[path_a, path_b]) as rec:
        ids = sorted(set(A) & set(B))
        if len(ids) < max(len(A), len(B)):
            print(f"note: comparing the {len(ids)} shared utterances (A {len(A)}, B {len(B)})")
        groups, ea, eb, words = [], [], [], []
        for u in ids:
            g, ref, ha = A[u]
            if normalize_bangla_text(ref) != normalize_bangla_text(B[u][1]):
                raise SystemExit(f"reference mismatch for {u}")
            a_e, n = word_edits(ref, ha)
            b_e, _ = word_edits(ref, B[u][2])
            groups.append(g), ea.append(a_e), eb.append(b_e), words.append(n)
        ea, eb, words = map(np.array, (ea, eb, words))
        diff = bootstrap_ratio(groups, eb - ea, words, n_boot=args.n_boot, seed=args.seed)
        res = {
            "a": args.a, "b": args.b, "n_utterances": len(ids), "n_groups": diff["n_groups"],
            "ref_words": int(words.sum()),
            "wer_a": float(ea.sum() / words.sum()), "wer_b": float(eb.sum() / words.sum()),
            "diff_b_minus_a": diff["value"], "diff_ci95": diff["ci95"],
            "p_b_better": float((diff["draws"] < 0).mean()),
            "n_boot": args.n_boot, "seed": args.seed, "mbr_temperature": args.mbr_temperature,
            "normalizer": "bangla_asr.common.text.normalize_bangla_text",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, json.dumps(res, ensure_ascii=False, indent=2))
        rec.outputs = [out]
    lo, hi = res["diff_ci95"]
    print(f"{args.name}: A {100 * res['wer_a']:.2f}%  B {100 * res['wer_b']:.2f}%  "
          f"B-A {100 * res['diff_b_minus_a']:+.2f} pts [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
          f"P(B better) {res['p_b_better']:.3f}  ({res['n_utterances']} utt, {res['n_groups']} groups)")


if __name__ == "__main__":
    main()
