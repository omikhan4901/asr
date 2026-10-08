"""
E5 / E21: what kinds of word errors a system makes, and which of them no
N-best rescorer can fix.

For each predictions.jsonl (pipeline/eval_zero_shot.py output), every word
edit is typed with bangla_asr.common.variants (number, script, segmentation,
convention, vowel_shift, near_miss, substitution, deletion, insertion). Type
counts add up exactly to the WER numerator. Reported per system, with 95%
cluster-bootstrap CIs over recordings/speakers:
  - share of edits per type;
  - variant share = segmentation + convention + vowel_shift (spelling/dialect
    variation), and the WER if those edits were not counted;
  - share of edits within 2 words of a chunk edge vs share of words there
    (alignment-boundary noise shows up as an excess at the edges).
With --nbest (one system only): c = number of N-best candidates that have the
reference word right, for each substitution/deletion; the type mix of
selection-proof errors (c = 0) vs fixable ones (c > 0). This is Table 2 and
the input to Gate 3.

Outputs in --out_dir/<set>/<model>/: errors.csv, summary.json, and with
--nbest the outputs go to --out_dir/with_nbest/<nbest stem>/ and include
by_agreement.csv. --all_zeroshot also writes
--out_dir/summary.csv (one row per system, E21).

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/error_types.py --all_zeroshot
  PYTHONPATH=src ./Scripts/python.exe analysis/error_types.py \
      --pred results/zeroshot/splits_fa-eval/<model>/predictions.jsonl --nbest data/nbest_fa/eval_nbest.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from bangla_asr.common.agreement import agreement_counts  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import TYPES, VARIANT_TYPES, classify_alignment  # noqa: E402

EDGE = 2  # words from either chunk end counted as "edge"


def analyse(pred_path: Path, out: Path, nbest_path=None, n_boot=1000):
    preds = read_jsonl(pred_path)
    nbest = {}
    if nbest_path:
        nbest = {r["utt_id"]: r for r in read_jsonl(nbest_path)}
        missing = sum(r["utt_id"] not in nbest for r in preds)
        if missing:
            print(f"  note: {missing} utterances have no N-best and get c = NaN")

    rows, per_utt = [], []
    for r in tqdm(preds, desc=pred_path.parent.name[:40], leave=False, mininterval=10):
        ref = normalize_bangla_text(r["ref"]).split()
        hyp = normalize_bangla_text(r["hyp"]).split()
        if not ref:
            continue
        edits = classify_alignment(ref, hyp)
        counts = None
        if r["utt_id"] in nbest:
            cands = sorted(nbest[r["utt_id"]]["candidates"], key=lambda c: c["rank"])
            counts = agreement_counts(ref, [normalize_bangla_text(c["text"]).split() for c in cands])
        n_edge = min(len(ref), 2 * EDGE)
        tc = Counter()
        for e in edits:
            i = e["ref_index"]
            if i is None:  # insertion: place it by hypothesis position
                i = round(e["hyp_index"] * len(ref) / max(1, len(hyp)))
            e["edge"] = i < EDGE or i >= len(ref) - EDGE
            e["c"] = int(counts[e["ref_index"]]) if counts is not None and e["ref_index"] is not None else np.nan
            e.update(utt_id=r["utt_id"], group=str(r["group"]))
            tc[e["type"]] += 1
            tc["edge"] += e["edge"]
        rows.extend(edits)
        per_utt.append({"group": str(r["group"]), "words": len(ref), "edits": len(edits),
                        "edge_words": n_edge, **tc})

    u = pd.DataFrame(per_utt).fillna(0)
    for t in TYPES + ["edge"]:
        if t not in u:
            u[t] = 0
    g, edits, words = u["group"].tolist(), u["edits"].values, u["words"].values
    variant = u[list(VARIANT_TYPES)].sum(1).values

    def ci(num, den):
        b = bootstrap_ratio(g, num, den, n_boot=n_boot)
        return {"value": b["value"], "ci95": b["ci95"]}

    summary = {
        "predictions": str(pred_path), "nbest": str(nbest_path) if nbest_path else None,
        "n_utterances": len(u), "n_groups": u["group"].nunique(), "ref_words": int(words.sum()),
        "edits": int(edits.sum()),
        "wer": ci(edits, words),
        "variant_share": ci(variant, edits),
        "wer_without_variants": ci(edits - variant, words),
        "type_share": {t: ci(u[t].values, edits) for t in TYPES},
        "edge_error_share": ci(u["edge"].values, edits),
        "edge_word_share": float(u["edge_words"].sum() / words.sum()),
    }
    out.mkdir(parents=True, exist_ok=True)
    cols = ["utt_id", "group", "op", "type", "ref_index", "hyp_index", "ref_word", "hyp_word", "edge", "c"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")

    if nbest_path:
        e = pd.DataFrame(rows)
        e = e[e["ref_index"].notna() & e["c"].notna()]
        e["agreement"] = np.where(e["c"] == 0, "selection_proof", "fixable")
        tab = e.groupby(["type", "agreement"]).size().unstack(fill_value=0).reindex(TYPES, fill_value=0)
        tab.to_csv(out / "by_agreement.csv", encoding="utf-8-sig")
        sp = e[e["c"] == 0]
        summary["selection_proof"] = {
            "ref_position_errors": int(len(e)), "selection_proof": int(len(sp)),
            "selection_proof_frac": float(len(sp) / max(1, len(e))),
            "variant_share_of_selection_proof": float(sp["type"].isin(VARIANT_TYPES).mean()) if len(sp) else None,
            "variant_share_of_fixable": float(e[e["c"] > 0]["type"].isin(VARIANT_TYPES).mean()),
            "type_share_of_selection_proof": {t: float((sp["type"] == t).mean()) for t in TYPES},
        }
        # Same shares with a cluster-bootstrap CI (resampling recordings).
        sp_u = sp.assign(v=sp["type"].isin(VARIANT_TYPES)).groupby("group")["v"].agg(["sum", "count"])
        summary["selection_proof"]["variant_share_of_selection_proof_ci"] = ci_from_groups(sp_u, n_boot)
    atomic_write_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def ci_from_groups(df, n_boot):
    b = bootstrap_ratio(df.index.tolist(), df["sum"].values, df["count"].values, n_boot=n_boot)
    return {"value": b["value"], "ci95": b["ci95"]}


def flat(s):
    row = {"set": Path(s["predictions"]).parent.parent.name, "model": Path(s["predictions"]).parent.name,
           "utts": s["n_utterances"], "WER%": 100 * s["wer"]["value"],
           "variant_share%": 100 * s["variant_share"]["value"],
           "variant_share_ci": "[%.1f, %.1f]" % tuple(100 * x for x in s["variant_share"]["ci95"]),
           "WER_wo_variants%": 100 * s["wer_without_variants"]["value"],
           "edge_err%": 100 * s["edge_error_share"]["value"], "edge_words%": 100 * s["edge_word_share"]}
    row.update({f"{t}%": 100 * s["type_share"][t]["value"] for t in TYPES})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", nargs="*", default=[])
    p.add_argument("--all_zeroshot", action="store_true", help="every results/zeroshot/*/*/predictions.jsonl")
    p.add_argument("--nbest", default=None, help="N-best jsonl for the c analysis (needs exactly one --pred)")
    p.add_argument("--out_dir", default="results/error_types")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    preds = [Path(x) for x in args.pred]
    if args.all_zeroshot:
        preds += sorted(Path("results/zeroshot").glob("*/*/predictions.jsonl"))
    if args.nbest and len(preds) != 1:
        raise SystemExit("--nbest needs exactly one --pred")
    if not preds:
        raise SystemExit("nothing to do: give --pred or --all_zeroshot")

    out_root = Path(args.out_dir)
    with RunRecord(out_root / ("run_info.json" if args.all_zeroshot else
                               f"{preds[0].parent.parent.name}__{preds[0].parent.name}"
                               f"{'__' + Path(args.nbest).stem if args.nbest else ''}.run_info.json"),
                   inputs=preds + ([args.nbest] if args.nbest else [])) as rec:
        summaries = []
        for pp in preds:
            out = out_root / pp.parent.parent.name / pp.parent.name
            if args.nbest:  # keep the plain per-system summary (E21) separate
                out = out_root / "with_nbest" / Path(args.nbest).stem
            s = analyse(pp, out, args.nbest, args.n_boot)
            summaries.append(s)
            sp = s.get("selection_proof")
            print(f"{pp.parent.parent.name:16s} {pp.parent.name[:45]:45s} WER {100 * s['wer']['value']:5.2f}  "
                  f"variant share {100 * s['variant_share']['value']:4.1f}% "
                  f"[{100 * s['variant_share']['ci95'][0]:.1f}, {100 * s['variant_share']['ci95'][1]:.1f}]"
                  + (f"  selection-proof {100 * sp['selection_proof_frac']:.1f}%, variant share of them "
                     f"{100 * sp['variant_share_of_selection_proof']:.1f}%" if sp else ""))
            rec.outputs += [out / "summary.json", out / "errors.csv"]
        if args.all_zeroshot:
            df = pd.DataFrame([flat(s) for s in summaries]).sort_values(["set", "WER%"])
            df.to_csv(out_root / "summary.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
            rec.outputs.append(out_root / "summary.csv")


if __name__ == "__main__":
    main()
