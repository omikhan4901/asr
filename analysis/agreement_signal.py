"""
E23: is candidate agreement a reference-free signal of where errors are, and
of what kind they are?

For every word j of a hypothesis (the greedy output by default), a_j = number
of N-best candidates that contain that word at that position (alignment of
each candidate to the hypothesis; no reference involved). Then, using the
reference only to score:
  - AUROC of (N - a_j) for "word j is wrong" (substitution or insertion), with
    a 95% cluster-bootstrap CI;
  - error rate per agreement level a = 0..N;
  - type mix per level: are spelling/dialect variants confident errors (all
    candidates agree on the wrong form) while acoustic errors are not?
  - flagging curve: flag words with a <= t; share of words flagged, of errors
    caught (recall) and of flags that are errors (precision), overall and for
    variant vs other errors. This bounds agreement-guided selective expansion
    (Phase 3). Deleted reference words have no hypothesis position and are
    counted separately.

Outputs in --out_dir/<name>/: summary.json, by_level.csv, flagging.csv.

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/agreement_signal.py \
      --pred results/zeroshot/splits_fa-eval/<model>/predictions.jsonl \
      --nbest data/nbest_fa/eval_nbest.jsonl --name eval_tugstugi_greedy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from bangla_asr.common.agreement import agreement_counts  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import VARIANT_TYPES, classify_alignment  # noqa: E402


def auroc_from_counts(err, ok):
    """AUROC of a score where lower level = more suspicious; err/ok are counts per level."""
    # P(level of a random error < level of a random correct word) + 0.5 * P(tie)
    ok_above = np.cumsum(ok[::-1])[::-1] - ok  # correct words at strictly higher levels
    pairs = err.sum() * ok.sum()
    return float(((err * ok_above).sum() + 0.5 * (err * ok).sum()) / pairs) if pairs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="predictions.jsonl of the hypothesis to score")
    p.add_argument("--nbest", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out_dir", default="results/agreement_signal")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir, args.name)
    out.mkdir(parents=True, exist_ok=True)
    with RunRecord(out / "run_info.json", inputs=[args.pred, args.nbest]) as rec:
        nbest = {r["utt_id"]: r for r in read_jsonl(args.nbest)}
        preds = [r for r in read_jsonl(args.pred) if r["utt_id"] in nbest]
        N = max(len(r["candidates"]) for r in nbest.values())
        rows, n_del = [], 0
        for r in tqdm(preds, desc="scoring", mininterval=10):
            ref = normalize_bangla_text(r["ref"]).split()
            hyp = normalize_bangla_text(r["hyp"]).split()
            if not ref or not hyp:
                continue
            cands = [normalize_bangla_text(c["text"]).split() for c in nbest[r["utt_id"]]["candidates"]]
            a = agreement_counts(hyp, cands)
            wrong = {}
            for e in classify_alignment(ref, hyp):
                if e["hyp_index"] is None:
                    n_del += 1
                else:
                    wrong[e["hyp_index"]] = e["type"]
            for j in range(len(hyp)):
                t = wrong.get(j, "correct")
                rows.append((str(r["group"]), int(a[j]), t != "correct", t))
        d = pd.DataFrame(rows, columns=["group", "a", "error", "type"])
        d["variant"] = d["type"].isin(VARIANT_TYPES)

        # Per group x level counts, so every bootstrap draw is a cheap sum.
        groups = d["group"].unique().tolist()
        gi = d["group"].map({g: k for k, g in enumerate(groups)}).values
        G, L = len(groups), N + 1

        def table(mask):
            t = np.zeros((G, L))
            np.add.at(t, (gi[mask], d["a"].values[mask]), 1)
            return t

        err_t, ok_t = table(d["error"].values), table(~d["error"].values)
        var_t = table((d["error"] & d["variant"]).values)
        samp = np.random.default_rng(args.seed).integers(0, G, size=(args.n_boot, G))
        aucs = [auroc_from_counts(err_t[s].sum(0), ok_t[s].sum(0)) for s in samp]
        err, ok, var = err_t.sum(0), ok_t.sum(0), var_t.sum(0)

        lvl = pd.DataFrame({"a": range(L), "words": err + ok, "errors": err, "variant_errors": var})
        lvl["error_rate"] = lvl["errors"] / lvl["words"].clip(lower=1)
        rates = np.array([err_t[s].sum(0) / np.maximum(1, (err_t[s] + ok_t[s]).sum(0)) for s in samp])
        lvl["error_rate_lo"], lvl["error_rate_hi"] = np.percentile(rates, 2.5, 0), np.percentile(rates, 97.5, 0)
        lvl["variant_share_of_errors"] = lvl["variant_errors"] / lvl["errors"].clip(lower=1)
        lvl.to_csv(out / "by_level.csv", index=False, encoding="utf-8-sig")

        flag = []
        for t in range(L):
            m = d["a"] <= t
            flag.append({"flag_if_a_le": t, "words_flagged": m.mean(),
                         "recall_all": (m & d["error"]).sum() / max(1, d["error"].sum()),
                         "recall_variant": (m & d["error"] & d["variant"]).sum() / max(1, (d["error"] & d["variant"]).sum()),
                         "recall_other": (m & d["error"] & ~d["variant"]).sum() / max(1, (d["error"] & ~d["variant"]).sum()),
                         "precision": (m & d["error"]).sum() / max(1, m.sum())})
        pd.DataFrame(flag).to_csv(out / "flagging.csv", index=False, float_format="%.4f", encoding="utf-8-sig")

        e = d[d["error"]]
        summary = {
            "pred": args.pred, "nbest": args.nbest, "n_candidates": N, "hyp_words": len(d),
            "hyp_errors": int(d["error"].sum()), "ref_deletions_not_scored": n_del, "n_groups": G,
            "auroc": auroc_from_counts(err, ok),
            "auroc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "mean_a_correct": float(d.loc[~d["error"], "a"].mean()),
            "mean_a_by_error_type": e.groupby("type")["a"].mean().round(3).to_dict(),
            "share_errors_full_agreement": float((e["a"] == N).mean()),
            "share_variant_errors_full_agreement": float((e.loc[e["variant"], "a"] == N).mean()),
            "share_other_errors_full_agreement": float((e.loc[~e["variant"], "a"] == N).mean()),
            "n_boot": args.n_boot, "seed": args.seed,
        }
        atomic_write_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        rec.outputs = [out / "summary.json", out / "by_level.csv", out / "flagging.csv"]
    print(f"{args.name}: AUROC {summary['auroc']:.3f} [{summary['auroc_ci95'][0]:.3f}, {summary['auroc_ci95'][1]:.3f}]; "
          f"errors with all {N} candidates agreeing: variant {100 * summary['share_variant_errors_full_agreement']:.1f}%, "
          f"other {100 * summary['share_other_errors_full_agreement']:.1f}%")


if __name__ == "__main__":
    main()
