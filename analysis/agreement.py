"""
E3: candidate-agreement anatomy of the N-best selection ceiling.

For every utterance and every reference word i, align each of the N
candidates to the reference and count c_i = how many candidates have word i
right (an "equal" alignment op). Then:

  - 1-best, MBR, and utterance-oracle WER (choose one whole candidate);
  - coverage miss rate: fraction of reference words with c_i = 0, i.e. no
    candidate has them right. No selection-based rescorer can fix these
    ("selection-proof"); it lower-bounds what any compositional combiner
    (ROVER-like) could reach, ignoring insertions;
  - histogram of the 1-best's reference-position errors (substitutions +
    deletions) over c = 0..N, and how many of each bin MBR fixed.

Outputs (in --out_dir): summary.json, histogram.csv, errors.csv (one row per
1-best reference-position error, for the error-type analysis E5).

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/agreement.py \
      --nbest data/nbest_fa/eval_nbest.jsonl --out_dir results/agreement/eval_tugstugi
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from bangla_asr.common.metrics import align_tokens  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.config import MBRConfig  # noqa: E402
from bangla_asr.rescoring.mbr import MBRSelector  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402


def correct_positions(ref, hyp):
    """Set of reference indices the hypothesis gets right, and the 1-best op per index."""
    ok, ops = set(), {}
    for o in align_tokens(ref, hyp):
        if o.ref_index is not None:
            ops[o.ref_index] = (o.op, o.hyp_token or "")
            if o.op == "equal":
                ok.add(o.ref_index)
    return ok, ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mbr_temperature", type=float, default=0.5)
    args = p.parse_args()

    recs = [json.loads(l) for l in open(args.nbest, encoding="utf-8") if l.strip()]
    selector = MBRSelector(MBRConfig(temperature=args.mbr_temperature))
    n_max = max(len(r["candidates"]) for r in recs)
    hist = np.zeros(n_max + 1, int)
    hist_mbr_fixed = np.zeros(n_max + 1, int)
    tot = {"ref_words": 0, "e_1best": 0, "e_mbr": 0, "e_oracle": 0, "coverage_miss": 0}
    err_rows = []

    for r in tqdm(recs, desc="aligning"):
        ref = normalize_bangla_text(r["ground_truth"]).split()
        if not ref:
            continue
        cands = sorted(r["candidates"], key=lambda c: c["rank"])
        toks = [normalize_bangla_text(c["text"]).split() for c in cands]
        mbr_text = selector.select(cands)["text"]
        mbr_toks = normalize_bangla_text(mbr_text).split()

        counts = np.zeros(len(ref), int)
        for t in toks:
            ok, _ = correct_positions(ref, t)
            for i in ok:
                counts[i] += 1
        ok1, ops1 = correct_positions(ref, toks[0])
        okm, _ = correct_positions(ref, mbr_toks)

        tot["ref_words"] += len(ref)
        tot["e_1best"] += Levenshtein.distance(ref, toks[0])
        tot["e_mbr"] += Levenshtein.distance(ref, mbr_toks)
        tot["e_oracle"] += min(Levenshtein.distance(ref, t) for t in toks)
        tot["coverage_miss"] += int((counts == 0).sum())
        for i, w in enumerate(ref):
            if i in ok1:
                continue
            c = int(counts[i])
            hist[c] += 1
            hist_mbr_fixed[c] += i in okm
            op, hw = ops1.get(i, ("delete", ""))
            err_rows.append({"utt_id": r["utt_id"], "group": r.get("group", ""), "ref_index": i,
                             "ref_word": w, "hyp_word": hw, "op": op, "c": c, "mbr_correct": i in okm})

    N = tot["ref_words"]
    n_err = int(hist.sum())
    summary = {
        "nbest": args.nbest, "n_utterances": len(recs), "n_candidates_max": n_max, "ref_words": N,
        "wer_1best": tot["e_1best"] / N, "wer_mbr": tot["e_mbr"] / N, "wer_utt_oracle": tot["e_oracle"] / N,
        "coverage_miss_rate": tot["coverage_miss"] / N,
        "mbr_gap_recovery": (tot["e_1best"] - tot["e_mbr"]) / max(1, tot["e_1best"] - tot["e_oracle"]),
        "onebest_ref_errors": n_err,
        "onebest_ref_errors_selection_proof_frac": float(hist[0] / max(1, n_err)),
        "mbr_temperature": args.mbr_temperature,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"c": range(n_max + 1), "onebest_errors": hist, "mbr_fixed": hist_mbr_fixed}).to_csv(
        out / "histogram.csv", index=False)
    pd.DataFrame(err_rows).to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")
    for k, v in summary.items():
        print(f"{k:42s} {v:.4f}" if isinstance(v, float) else f"{k:42s} {v}")


if __name__ == "__main__":
    main()
