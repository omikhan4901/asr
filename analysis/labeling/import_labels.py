"""
Download the team's labels from Supabase and compute the Phase 2 label results.

Saves a dated snapshot of items + labels (results/labels/snapshot_<time>.json)
and computes from it (or from --snapshot, offline):

  refs    agreement on double-labeled chunks (status agreement, WER between the
          two corrected texts); reference noise per source (E6): word edit rate
          original -> verified, typed with the same rules (spelling / dialect /
          content); every zero-shot model's WER on the verified chunks against
          the original vs the verified reference, paired cluster bootstrap.
  audit   judgment shares over all differences (model wrong / transcript wrong /
          both wrong / spelling / dialect / can't tell), by rule type, and split
          by selection-proof (no N-best candidate has the reference words, c = 0)
          vs fixable, with cluster-bootstrap CIs over recordings. Items labeled by
          several people count each label 1/n. Fleiss kappa on the overlap units.
  pairs   Fleiss / pairwise Cohen kappa on the overlap pairs; human label shares
          per rule type (unweighted and weighted by pair frequency).

Outputs: results/labels/summary.json, verified_refs.csv, audit_units.csv, pair_labels.csv.

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/import_labels.py
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/import_labels.py --snapshot results/labels/snapshot_X.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from bangla_asr.common.agreement import agreement_counts  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import classify_alignment  # noqa: E402
from build_tasks import read_env  # noqa: E402

AUDIT = ["model", "ref", "both", "spelling", "dialect", "unsure"]
PAIR = ["spelling", "dialect", "different", "unsure"]
SET_OF_SOURCE = {"dlsprint": "splits_fa-eval", "fleurs": "fleurs-test", "kathbath": "kathbath-valid",
                 "commonvoice": "cv-test"}


def words(t):
    return normalize_bangla_text(t).split()


PRIMARY_KEY = {"items": "id", "labels": "item_id,annotator", "resolutions": "item_id"}  # stable paging order


def fetch_all(url, key, table, select):
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, start = [], 0
    while True:
        r = requests.get(f"{url}/rest/v1/{table}?select={select}&order={PRIMARY_KEY[table]}", timeout=60,
                         headers={**h, "Range": f"{start}-{start + 999}"})
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < 1000:
            return rows
        start += 1000


def fleiss_kappa(table):
    table = np.asarray(table, float)
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def cohen_kappa(a, b):
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def shares_ci(df, value_col, cats, group_col, weight_col="w", n_boot=1000):
    """Share of each category (weighted rows) with a cluster-bootstrap 95% CI."""
    out = {}
    for c in cats:
        b = bootstrap_ratio(df[group_col].tolist(), (df[value_col] == c) * df[weight_col], df[weight_col], n_boot=n_boot)
        out[c] = {"value": b["value"], "ci95": b["ci95"]}
    return out


def refs_results(items, labels, res, n_boot):
    ref_items = {i["id"]: i for i in items if i["task"] == "ref"}
    L = [l for l in labels if l["item_id"] in ref_items and l["status"] == "done"]
    rows = []
    for l in L:
        it, d = ref_items[l["item_id"]], l["data"]
        # Punctuation-only edits are not corrections: scoring strips punctuation.
        status = "correct" if d["status"] == "corrected" and words(d["text"]) == words(it["payload"]["text"]) else d["status"]
        rows.append({"id": it["id"], "annotator": l["annotator"], "source": it["meta"]["source"],
                     "utt_id": it["meta"]["utt_id"], "group": it["meta"]["group"], "status": status,
                     "text_original": it["payload"]["text"], "text_verified": d["text"],
                     "cut_off": d.get("cut_off", False), "note": d.get("note", ""), "updated_at": l["updated_at"]})
    R = pd.DataFrame(rows)
    res = {"labels": len(R)}
    if R.empty:
        return R, res
    R = R.sort_values("updated_at")
    R["final"] = ~R.duplicated("id", keep="first")  # first finished label, unless the team agreed on one
    agreed = []
    for iid, d in res.items():
        if iid in ref_items:
            it = ref_items[iid]
            agreed.append({"id": iid, "annotator": "_agreed", "source": it["meta"]["source"], "utt_id": it["meta"]["utt_id"],
                           "group": it["meta"]["group"], "status": d["status"], "text_original": it["payload"]["text"],
                           "text_verified": d["text"], "cut_off": d.get("cut_off", False), "note": "", "updated_at": "",
                           "final": True})
    if agreed:
        R.loc[R["id"].isin(res), "final"] = False
        R = pd.concat([R, pd.DataFrame(agreed)], ignore_index=True)
    pairs = [g for _, g in R[R["annotator"] != "_agreed"].groupby("id") if len(g) > 1]
    if pairs:
        ed = n = 0
        for g in pairs:
            a, b = g.iloc[0], g.iloc[1]
            if "unusable" not in (a["status"], b["status"]):
                ed += Levenshtein.distance(words(a["text_verified"]), words(b["text_verified"]))
                n += len(words(a["text_verified"]))
        res["agreement"] = {"double_labeled": len(pairs),
                            "status_agreement": float(np.mean([g.iloc[0]["status"] == g.iloc[1]["status"] for g in pairs])),
                            "inter_annotator_wer": ed / n if n else None}
    F = R[R["final"]]
    res["by_source"] = {}
    for src, S in F.groupby("source"):
        U = S[S["status"] != "unusable"]
        kinds, n_words, per_utt = Counter(), 0, []
        for _, r in U.iterrows():
            o, v = words(r["text_original"]), words(r["text_verified"])
            k = Counter("spelling" if e["type"] == "convention" else "dialect" if e["type"] == "vowel_shift" else "content"
                        for e in classify_alignment(o, v))
            kinds.update(k)
            n_words += len(o)
            per_utt.append((r["group"], sum(k.values()), len(o)))
        g = [x[0] for x in per_utt]
        noise = bootstrap_ratio(g, [x[1] for x in per_utt], [x[2] for x in per_utt], n_boot=n_boot) if per_utt else None
        entry = {"chunks": len(S), "unusable": int((S["status"] == "unusable").sum()),
                 "corrected_share": float((U["status"] == "corrected").mean()) if len(U) else None,
                 "cut_off_share": float(S["cut_off"].mean()),
                 "edit_rate": {"value": noise["value"], "ci95": noise["ci95"]} if noise else None,
                 "edits_by_kind": dict(kinds), "original_words": n_words, "models": {}}
        ver = {r["utt_id"]: r["text_verified"] for _, r in U.iterrows()}
        grp = {r["utt_id"]: r["group"] for _, r in U.iterrows()}
        for pp in sorted(Path("results/zeroshot", SET_OF_SOURCE[src]).glob("*/predictions.jsonl")):
            g, eo, wo, ev, wv = [], [], [], [], []
            for r in read_jsonl(pp):
                if r["utt_id"] in ver:
                    h, o, v = words(r["hyp"]), words(r["ref"]), words(ver[r["utt_id"]])
                    g.append(grp[r["utt_id"]])
                    eo.append(Levenshtein.distance(o, h)), wo.append(len(o))
                    ev.append(Levenshtein.distance(v, h)), wv.append(len(v))
            if not g:
                continue
            uniq = {x: k for k, x in enumerate(dict.fromkeys(g))}
            gi, G = np.array([uniq[x] for x in g]), len(uniq)
            sums = [np.bincount(gi, np.asarray(x, float), G) for x in (eo, wo, ev, wv)]
            samp = np.random.default_rng(0).integers(0, G, size=(n_boot, G))
            diff = sums[2][samp].sum(1) / sums[3][samp].sum(1) - sums[0][samp].sum(1) / sums[1][samp].sum(1)
            entry["models"][pp.parent.name] = {
                "utts": len(g), "wer_original": sum(eo) / sum(wo), "wer_verified": sum(ev) / sum(wv),
                "diff_verified_minus_original": sum(ev) / sum(wv) - sum(eo) / sum(wo),
                "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        res["by_source"][src] = entry
    return R, res


def audit_results(items, labels, res, nbest_path, n_boot):
    aud = {i["id"]: i for i in items if i["task"] == "audit"}
    L = [l for l in labels if l["item_id"] in aud and l["status"] == "done"]
    nbest = {r["utt_id"]: r for r in read_jsonl(nbest_path)} if Path(nbest_path).exists() else {}
    by_item = defaultdict(list)
    for l in L:
        by_item[l["item_id"]].append(l)
    rows = []
    for iid, ls in by_item.items():
        it = aud[iid]
        p, m = it["payload"], it["meta"]
        c = None
        if m["utt_id"] in nbest:
            cands = [words(x["text"]) for x in nbest[m["utt_id"]]["candidates"]]
            c = agreement_counts(p["ref"], cands)
        for k, u in enumerate(p["units"]):
            if c is not None and u["r1"] > u["r0"]:
                sp = "selection_proof" if int(c[u["r0"]:u["r1"]].max()) == 0 else "fixable"
            else:
                sp = "insertion" if u["r1"] == u["r0"] else "unknown"
            # Agreed answer (if the team settled this item) carries the weight; independent labels keep
            # weight 0 then, but stay in the table for kappa.
            votes = [(l["annotator"], l["data"]["units"][k], 0.0 if iid in res else 1 / len(ls)) for l in ls]
            if iid in res:
                votes.append(("_agreed", res[iid]["units"][k], 1.0))
            for who, judgment, w in votes:
                rows.append({"item": iid, "unit": k, "annotator": who, "judgment": judgment,
                             "w": w, "group": m["group"], "rule_type": m["unit_types"][k],
                             "edits": m["unit_edits"][k], "agreement": sp,
                             "ref_span": " ".join(p["ref"][u["r0"]:u["r1"]]), "hyp_span": " ".join(p["hyp"][u["h0"]:u["h1"]])})
    A = pd.DataFrame(rows)
    res = {"labels": len(L), "units": int(A[["item", "unit"]].drop_duplicates().shape[0]) if len(A) else 0}
    if A.empty:
        return A, res
    res["shares"] = shares_ci(A, "judgment", AUDIT, "group", n_boot=n_boot)
    res["shares_by_agreement"] = {g: shares_ci(D, "judgment", AUDIT, "group", n_boot=n_boot)
                                  for g, D in A.groupby("agreement") if len(D) >= 20}
    res["shares_by_rule_type"] = {t: {j: float((D["judgment"] == j).mul(D["w"]).sum() / D["w"].sum()) for j in AUDIT}
                                  for t, D in A.groupby("rule_type")}
    # Not a model error = transcript wrong, or the same word written differently.
    A["not_model_error"] = A["judgment"].isin(["ref", "spelling", "dialect"])
    res["not_model_error_share_by_agreement"] = {
        g: (lambda b: {"value": b["value"], "ci95": b["ci95"]})(
            bootstrap_ratio(D["group"].tolist(), D["not_model_error"] * D["w"], D["w"], n_boot=n_boot))
        for g, D in A.groupby("agreement") if len(D) >= 20}
    indep = A[A["annotator"] != "_agreed"]
    multi = indep.groupby(["item", "unit"]).filter(lambda g: g["annotator"].nunique() >= 3)
    if len(multi):
        tab = multi.pivot_table(index=["item", "unit"], columns="judgment", values="w", aggfunc="count", fill_value=0)
        tab = tab.reindex(columns=AUDIT, fill_value=0)
        tab = tab[tab.sum(axis=1) == tab.sum(axis=1).max()]
        res["fleiss_kappa"] = {"units": len(tab), "kappa": fleiss_kappa(tab.values)}
    return A, res


def pair_results(items, labels, res):
    pit = {i["id"]: i for i in items if i["task"] == "pair"}
    rows = [{"id": l["item_id"], "annotator": l["annotator"], "label": l["data"]["label"], "note": l["data"].get("note", ""),
             "ref_word": pit[l["item_id"]]["payload"]["ref_word"], "hyp_word": pit[l["item_id"]]["payload"]["hyp_word"],
             **{k: pit[l["item_id"]]["meta"][k] for k in ("rule_type", "stratum", "model", "count")},
             "target": pit[l["item_id"]]["target"]}
            for l in labels if l["item_id"] in pit and l["status"] == "done"]
    P = pd.DataFrame(rows)
    res = {"labels": len(P)}
    if P.empty:
        return P, res
    ov = P[P["target"] > 1]  # independent labels only (agreed answers are added below)
    names = sorted(ov["annotator"].unique())
    full = ov.groupby("id").filter(lambda g: g["annotator"].nunique() == len(names))
    if len(names) >= 2 and len(full):
        tab = full.pivot_table(index="id", columns="label", values="annotator", aggfunc="count", fill_value=0)
        wide = full.pivot(index="id", columns="annotator", values="label")
        res["agreement"] = {"pairs": int(full["id"].nunique()), "annotators": names,
                            "fleiss_kappa": fleiss_kappa(tab.reindex(columns=PAIR, fill_value=0).values),
                            "cohen_kappa": {f"{a}~{b}": cohen_kappa(list(wide[a]), list(wide[b])) for a, b in combinations(names, 2)}}
    P["w"] = np.where(P["id"].isin(res), 0.0, 1 / P.groupby("id")["id"].transform("count"))
    agreed = [{"id": iid, "annotator": "_agreed", "label": d["label"], "note": "", "w": 1.0,
               "ref_word": pit[iid]["payload"]["ref_word"], "hyp_word": pit[iid]["payload"]["hyp_word"],
               **{k: pit[iid]["meta"][k] for k in ("rule_type", "stratum", "model", "count")}, "target": pit[iid]["target"]}
              for iid, d in res.items() if iid in pit]
    if agreed:
        P = pd.concat([P, pd.DataFrame(agreed)], ignore_index=True)
    res["rule_vs_human"] = {}
    for t, D in P.groupby("rule_type"):
        res["rule_vs_human"][t] = {"pairs": int(D["id"].nunique()),
                                   "share": {j: float(D.loc[D["label"] == j, "w"].sum() / D["w"].sum()) for j in PAIR},
                                   "share_weighted_by_count": {j: float((D["w"] * D["count"])[D["label"] == j].sum() / (D["w"] * D["count"]).sum()) for j in PAIR}}
    return P, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default=None, help="use a saved snapshot instead of downloading")
    p.add_argument("--nbest", default="data/nbest_fa/eval_nbest.jsonl", help="for selection-proof vs fixable (texts only)")
    p.add_argument("--out_dir", default="results/labels")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snap_path = Path(args.snapshot)
    else:
        env = read_env()
        url, key = env["SUPABASE_URL"].rstrip("/"), env["SUPABASE_SECRET_KEY"]
        snap = {"downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "items": fetch_all(url, key, "items", "*"),
                "labels": fetch_all(url, key, "labels", "*"),
                "resolutions": fetch_all(url, key, "resolutions", "*")}
        snap_path = out / f"snapshot_{time.strftime('%Y%m%d_%H%M')}.json"
        atomic_write_text(snap_path, json.dumps(snap, ensure_ascii=False))
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    with RunRecord(out / "run_info.json", inputs=[snap_path, args.nbest]) as rec:
        res = {r["item_id"]: r["data"] for r in snap.get("resolutions", [])}
        R, refs = refs_results(snap["items"], snap["labels"], res, args.n_boot)
        A, audit = audit_results(snap["items"], snap["labels"], res, args.nbest, args.n_boot)
        P, pairs = pair_results(snap["items"], snap["labels"], res)
        summary_res = {"resolutions": len(res)}
        summary = {"snapshot": str(snap_path), **summary_res, "refs": refs, "audit": audit, "pairs": pairs}
        atomic_write_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        R.to_csv(out / "verified_refs.csv", index=False, encoding="utf-8-sig")
        A.to_csv(out / "audit_units.csv", index=False, encoding="utf-8-sig")
        P.to_csv(out / "pair_labels.csv", index=False, encoding="utf-8-sig")
        rec.outputs = [out / "summary.json", out / "verified_refs.csv", out / "audit_units.csv", out / "pair_labels.csv"]
    print(f"refs {refs['labels']} labels, audit {audit['labels']} labels / {audit['units']} units, pairs {pairs['labels']} labels "
          f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
