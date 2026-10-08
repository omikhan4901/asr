"""
Read the annotators' label files back and compute the Phase 2 label results.

Put every returned labels_<name>.json in --labels_dir (default
labeling_kit/returned/). Uses labeling_kit/manifest.json for what was sampled.

Outputs in --out_dir (default results/labels/):
  verified_refs.csv     one row per labeled chunk and annotator; final = the
                        chunk owner's label (the second label of an overlap
                        chunk is only used for agreement)
  pair_labels.csv       one row per pair and annotator, with the hidden rule type
  summary.json:
    refs_agreement      overlap chunks: status agreement, WER between the two
                        annotators' texts
    pairs_agreement     overlap pairs: Fleiss kappa (all annotators), pairwise
                        Cohen kappa
    rule_vs_human       per rule type: share of human labels spelling / dialect /
                        different / unsure (majority label per pair), unweighted
                        and weighted by pair frequency
    reference_noise     E6: word edit rate original -> verified reference, typed
                        with the same rules (convention = spelling, vowel_shift =
                        dialect written as spoken, other = content), and every
                        model's WER on the verified chunks against the original
                        and the verified references (paired cluster bootstrap)

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/import_labels.py
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import classify_alignment  # noqa: E402

LABELS = ["spelling", "dialect", "different", "unsure"]


def words(t):
    return normalize_bangla_text(t).split()


def fleiss_kappa(table):
    """table: items x categories counts, same number of raters per item."""
    table = np.asarray(table, float)
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kit_dir", default="labeling_kit")
    p.add_argument("--labels_dir", default="labeling_kit/returned")
    p.add_argument("--out_dir", default="results/labels")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    man = json.loads(Path(args.kit_dir, "manifest.json").read_text(encoding="utf-8"))
    files = sorted(Path(args.labels_dir).glob("labels_*.json"))
    if not files:
        raise SystemExit(f"no labels_*.json in {args.labels_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with RunRecord(out / "run_info.json", inputs=[Path(args.kit_dir, "manifest.json")] + files) as rec:
        labels = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        owner = {}
        for name, ids in man["ref_assignment"].items():
            for i in ids:
                owner.setdefault(i, name)  # first assignment = owner (overlap copies come after)
        ref_meta = {r["id"]: r for r in man["refs"]}
        pair_meta = {x["id"]: x for x in man["pairs"]}

        # ---- references
        rrows = []
        for L in labels:
            for i, v in L["refs"].items():
                if not v.get("status"):
                    continue
                m = ref_meta[i]
                rrows.append({"id": i, "utt_id": m["utt_id"], "recording": m["recording"],
                              "annotator": L["annotator"], "final": owner[i] == L["annotator"],
                              "status": v["status"], "text_original": m["text"],
                              "text_verified": v["text"] if v["status"] != "correct" else m["text"],
                              "cut_off": v.get("cut_off", False), "note": v.get("note", "")})
        R = pd.DataFrame(rrows)
        R.to_csv(out / "verified_refs.csv", index=False, encoding="utf-8-sig")
        summary = {"annotators": [L["annotator"] for L in labels],
                   "refs_labeled": int(R["final"].sum()) if len(R) else 0, "refs_total": len(man["refs"])}

        dup = R[R["id"].duplicated(keep=False)] if len(R) else R
        if len(dup):
            same_status, ed, n = [], 0, 0
            for _, g in dup.groupby("id"):
                if len(g) < 2:
                    continue
                a, b = g.iloc[0], g.iloc[1]
                same_status.append(a["status"] == b["status"])
                if "unusable" not in (a["status"], b["status"]):
                    ed += Levenshtein.distance(words(a["text_verified"]), words(b["text_verified"]))
                    n += len(words(a["text_verified"]))
            summary["refs_agreement"] = {"overlap_chunks": len(same_status),
                                         "status_agreement": float(np.mean(same_status)),
                                         "inter_annotator_wer": ed / n if n else None}

        # ---- E6: reference noise on the final, usable chunks
        F = R[R["final"] & (R["status"] != "unusable")] if len(R) else R
        if len(F):
            noise = Counter()
            n_words = 0
            for _, r in F.iterrows():
                o, v = words(r["text_original"]), words(r["text_verified"])
                n_words += len(o)
                for e in classify_alignment(o, v):
                    t = e["type"]
                    noise["spelling" if t == "convention" else "dialect" if t == "vowel_shift" else "content"] += 1
            summary["reference_noise"] = {
                "chunks": len(F), "unusable": int((R["final"] & (R["status"] == "unusable")).sum()),
                "corrected_share": float((F["status"] == "corrected").mean()),
                "original_words": n_words,
                "edit_rate_original_to_verified": sum(noise.values()) / max(1, n_words),
                "edits_by_kind": dict(noise),
            }
            ver = dict(zip(F["utt_id"], F["text_verified"]))
            grp = dict(zip(F["utt_id"], F["recording"]))
            models = {}
            for pp in sorted(Path("results/zeroshot/splits_fa-eval").glob("*/predictions.jsonl")):
                g, eo, ev, wo, wv = [], [], [], [], []
                for r in read_jsonl(pp):
                    if r["utt_id"] not in ver:
                        continue
                    h, o, v = words(r["hyp"]), words(r["ref"]), words(ver[r["utt_id"]])
                    g.append(grp[r["utt_id"]])
                    eo.append(Levenshtein.distance(o, h)), wo.append(len(o))
                    ev.append(Levenshtein.distance(v, h)), wv.append(len(v))
                if not g:
                    continue
                bo = bootstrap_ratio(g, eo, wo, n_boot=args.n_boot)
                bv = bootstrap_ratio(g, ev, wv, n_boot=args.n_boot)
                # Paired difference of two ratios with different denominators: resample groups jointly.
                uniq = {x: k for k, x in enumerate(dict.fromkeys(g))}
                gi = np.array([uniq[x] for x in g])
                G = len(uniq)
                sums = [np.bincount(gi, np.asarray(x, float), G) for x in (eo, wo, ev, wv)]
                samp = np.random.default_rng(0).integers(0, G, size=(args.n_boot, G))
                d = sums[2][samp].sum(1) / sums[3][samp].sum(1) - sums[0][samp].sum(1) / sums[1][samp].sum(1)
                models[pp.parent.name] = {"utts": len(g), "wer_original": bo["value"], "wer_original_ci95": bo["ci95"],
                                          "wer_verified": bv["value"], "wer_verified_ci95": bv["ci95"],
                                          "diff_verified_minus_original": bv["value"] - bo["value"],
                                          "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            summary["reference_noise"]["models"] = models

        # ---- word pairs
        prow = []
        for L in labels:
            for i, v in L["pairs"].items():
                if v.get("label"):
                    m = pair_meta[i]
                    prow.append({"id": i, "annotator": L["annotator"], "label": v["label"], "note": v.get("note", ""),
                                 "ref_word": m["ref_word"], "hyp_word": m["hyp_word"], "count": m["count"],
                                 "rule_type": m["rule_type"], "stratum": m["stratum"]})
        P = pd.DataFrame(prow)
        P.to_csv(out / "pair_labels.csv", index=False, encoding="utf-8-sig")
        if len(P):
            ov = P[P["id"].isin(man["pair_overlap_ids"])]
            names = sorted(ov["annotator"].unique())
            full = ov.groupby("id").filter(lambda g: g["annotator"].nunique() == len(names))
            pa = {"overlap_pairs_complete": int(full["id"].nunique()), "annotators": names}
            if len(names) >= 2 and len(full):
                tab = full.pivot_table(index="id", columns="label", values="annotator", aggfunc="count", fill_value=0)
                pa["fleiss_kappa"] = fleiss_kappa(tab.reindex(columns=LABELS, fill_value=0).values)
                wide = full.pivot(index="id", columns="annotator", values="label")
                pa["cohen_kappa"] = {f"{a}~{b}": cohen_kappa(list(wide[a]), list(wide[b]))
                                     for a, b in combinations(names, 2)}
            summary["pairs_agreement"] = pa
            maj = P.groupby("id").agg(label=("label", lambda s: s.mode().iloc[0] if s.mode().size == 1 else "unsure"),
                                      rule_type=("rule_type", "first"), count=("count", "first"),
                                      stratum=("stratum", "first"))
            rvh = {}
            for t, g in maj.groupby("rule_type"):
                rvh[t] = {"pairs": len(g),
                          "share": {lab: float((g["label"] == lab).mean()) for lab in LABELS},
                          "share_weighted": {lab: float(g.loc[g["label"] == lab, "count"].sum() / g["count"].sum())
                                             for lab in LABELS}}
            summary["rule_vs_human"] = rvh
            top = maj[maj["stratum"] == "top"]
            summary["top_pairs_variant_share_weighted"] = float(
                top.loc[top["label"].isin(["spelling", "dialect"]), "count"].sum() / max(1, top["count"].sum()))
        atomic_write_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        rec.outputs = [out / "summary.json", out / "verified_refs.csv", out / "pair_labels.csv"]
    print(json.dumps({k: v for k, v in summary.items() if k != "reference_noise"}, ensure_ascii=False, indent=1))
    if "reference_noise" in summary:
        rn = summary["reference_noise"]
        print(f"reference noise: {100 * rn['edit_rate_original_to_verified']:.2f}% of words edited {rn['edits_by_kind']}")
        for m, v in rn.get("models", {}).items():
            print(f"  {m[:50]:50s} WER original {100 * v['wer_original']:.2f}  verified {100 * v['wer_verified']:.2f}  "
                  f"diff {100 * v['diff_verified_minus_original']:+.2f} [{100 * v['diff_ci95'][0]:+.2f}, {100 * v['diff_ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
