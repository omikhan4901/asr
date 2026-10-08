"""
Build the shared labeling tasks and upload them to Supabase (Phase 2, E5/E6).

Three tasks, every item with a priority (lower = served first) and a target
number of annotators (> 1 = agreement item, served first within its block):

  ref    Verified references (E6). Listen, correct the transcript. No model
         output shown. Blocks in serving order:
           1. 300 DL Sprint FA eval chunks (stratified by recording)
           2. 100 each from FLEURS test, Kathbath valid, Common Voice test
           3. 300 more DL Sprint eval chunks (stratified)
         10% of each block is labeled by 2 annotators.
  audit  Error audit (Gate 3). DL Sprint eval chunks not in the ref sample on
         which the main model's greedy output has 1-15 differences; every
         difference (one alignment edit, or one re-spacing span) is judged:
         model wrong / transcript wrong / both wrong / same word spelled
         differently / dialect vs standard / can't tell. 400 chunks, stratified
         by recording; the first 60 are labeled by all 3.
  pair   Word pairs (rule precision). Substitution pairs ref word -> output word:
         the 600 most frequent of the main model over eval + FLEURS + Kathbath,
         the 50 most frequent remaining per rule type, and the 200 most frequent
         remaining pairs of the other Bangla models; 150 random pairs labeled by
         all 3.

Hidden from annotators (stored in items.meta): source, rule types, model,
counts. Sampling is seeded; re-running produces the same items and only
uploads what is missing (idempotent).

Needs SUPABASE_URL and SUPABASE_SECRET_KEY in .env (repo root, never committed).

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/build_tasks.py --dry_run   # build locally only
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/build_tasks.py             # build + upload
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, "src")
from bangla_asr.common.audio import load_row_audio  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import align_words, classify_alignment  # noqa: E402

MAIN = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
OTHERS = ["ai4bharat__indicwav2vec_v1_bengali", "bangla-speech-processing__BanglaASR",
          "arijitx__wav2vec2-xls-r-300m-bengali"]
PAIR_SETS = ["splits_fa-eval", "fleurs-test", "kathbath-valid"]
PAIR_TYPES = ["convention", "vowel_shift", "near_miss", "substitution"]
CACHE = Path("labeling_cache")


def stratified(df, n, key, seed):
    """n rows, allocated to groups of `key` in proportion to their size (largest remainder)."""
    sizes = df.groupby(key).size()
    quota = sizes / sizes.sum() * n
    alloc = quota.astype(int)
    for g in (quota - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[g] += 1
    parts = [df[df[key] == g].sample(min(k, sizes[g]), random_state=seed) for g, k in alloc.items() if k]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def words(t):
    return normalize_bangla_text(t).split()


def ref_items(seed):
    eval_df = pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig")
    dl = stratified(eval_df, 600, "recording", seed)
    dl_core, dl_extra = dl.iloc[0::2], dl.iloc[1::2]  # both halves stay spread over recordings
    others = []
    for name, path in [("fleurs", "data/external/fleurs/test.csv"), ("kathbath", "data/external/kathbath/valid.csv"),
                       ("commonvoice", "data/external/cv/test.csv")]:
        d = pd.read_csv(path, encoding="utf-8-sig")
        d = d.drop_duplicates("group") if name == "fleurs" else d  # FLEURS: one reading per sentence
        others.append(stratified(d.assign(source=name), 100, "group", seed) if name == "kathbath"
                      else d.assign(source=name).sample(100, random_state=seed))
    blocks = [dl_core.assign(source="dlsprint"), pd.concat(others).sample(frac=1, random_state=seed),
              dl_extra.assign(source="dlsprint")]
    items = []
    for b, block in enumerate(blocks):
        n_overlap = round(0.1 * len(block))
        for k, (_, r) in enumerate(block.iterrows()):
            items.append({"row": r, "target": 2 if k < n_overlap else 1, "block": b,
                          "payload": {"text": str(r["text"]), "duration": round(float(r["duration"]), 1)},
                          "meta": {"source": r["source"], "utt_id": r["utt_id"],
                                   "group": str(r.get("recording", r.get("group", ""))), "block": b}})
    return items


def audit_units(ref, hyp):
    """Alignment differences as display units: each edit, or one unit per re-spacing span."""
    ops = align_words(ref, hyp)
    typed = classify_alignment(ref, hyp)  # same alignment, edits in order
    units, i, j, t = [], 0, 0, 0
    for o in ops:
        if o.op == "equal":
            i, j = i + 1, j + 1
            continue
        e = typed[t]
        t += 1
        r1, h1 = i + (o.ref_token is not None), j + (o.hyp_token is not None)
        last = units[-1] if units else None
        if e["type"] == "segmentation" and last and last["type"] == "segmentation" and last["r1"] == i and last["h1"] == j:
            last["r1"], last["h1"] = r1, h1
            last["edits"] += 1
        else:
            units.append({"r0": i, "r1": r1, "h0": j, "h1": h1, "type": e["type"], "edits": 1})
        i, j = r1, h1
    return units


def audit_items(exclude, seed, n=400, n_overlap=60, max_units=15):
    eval_df = pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig").set_index("utt_id")
    preds = read_jsonl(Path("results/zeroshot/splits_fa-eval", MAIN, "predictions.jsonl"))
    rows, n_err, n_long = [], 0, 0
    for p in preds:
        if p["utt_id"] in exclude:
            continue
        ref, hyp = words(p["ref"]), words(p["hyp"])
        units = audit_units(ref, hyp)
        if not units:
            continue
        n_err += 1
        if len(units) > max_units:
            n_long += 1
            continue
        rows.append({"utt_id": p["utt_id"], "recording": p["group"], "ref": ref, "hyp": hyp, "units": units})
    pool = pd.DataFrame(rows)
    pick = stratified(pool, n, "recording", seed)
    items = []
    for k, (_, r) in enumerate(pick.iterrows()):
        row = eval_df.loc[r["utt_id"]].copy()
        row["utt_id"] = r["utt_id"]
        items.append({"row": row, "target": 3 if k < n_overlap else 1, "block": 0,
                      "payload": {"ref": r["ref"], "hyp": r["hyp"], "duration": round(float(row["duration"]), 1),
                                  "units": [{x: u[x] for x in ("r0", "r1", "h0", "h1")} for u in r["units"]]},
                      "meta": {"source": "dlsprint", "utt_id": r["utt_id"], "group": r["recording"], "model": MAIN,
                               "unit_types": [u["type"] for u in r["units"]],
                               "unit_edits": [u["edits"] for u in r["units"]]}})
    info = {"chunks_with_errors": n_err, "dropped_over_max_units": n_long, "max_units": max_units}
    return items, info


def collect_pairs(model, rng):
    """Substitution pairs of one model over PAIR_SETS: counts, rule type, 2 random examples each."""
    counts, types, examples = Counter(), {}, defaultdict(list)
    for s in PAIR_SETS:
        ep = Path("results/error_types", s, model, "errors.csv")
        if not ep.exists():
            continue
        e = pd.read_csv(ep, encoding="utf-8-sig", dtype={"ref_word": str, "hyp_word": str})
        e = e[(e["op"] == "substitute") & (e["type"] != "segmentation")]
        for r in e.itertuples():
            key = (r.ref_word, r.hyp_word)
            counts[key] += 1
            types[key] = r.type
            ex, item = examples[key], (model, s, r.utt_id, int(r.ref_index), int(r.hyp_index))
            if len(ex) < 2:
                ex.append(item)
            elif rng.random() < 2 / counts[key]:  # reservoir sample
                ex[rng.randrange(2)] = item
    return counts, types, examples


def pair_items(seed, n_top=600, n_per_type=50, n_other=200, n_overlap=150):
    rng = random.Random(seed)
    main_c, main_t, main_ex = collect_pairs(MAIN, rng)
    ranked = [k for k, _ in main_c.most_common()]
    chosen = {k: "top" for k in ranked[:n_top]}
    for t in PAIR_TYPES:
        chosen.update({k: f"type:{t}" for k in [k for k in ranked if k not in chosen and main_t[k] == t][:n_per_type]})
    oth_c, oth_t, oth_ex = Counter(), {}, {}
    for m in OTHERS:
        c, t, ex = collect_pairs(m, rng)
        oth_c.update(c)
        oth_t.update({k: v for k, v in t.items() if k not in oth_t})
        oth_ex.update({k: v for k, v in ex.items() if k not in oth_ex})
    chosen.update({k: "other_models" for k, _ in [kv for kv in oth_c.most_common() if kv[0] not in chosen][:n_other]})

    preds = {}

    def context(model, s, utt, ri, hi):
        if (model, s) not in preds:
            preds[model, s] = {r["utt_id"]: r for r in read_jsonl(Path("results/zeroshot", s, model, "predictions.jsonl"))}
        r = preds[model, s][utt]
        return {"ref": words(r["ref"]), "ref_i": ri, "hyp": words(r["hyp"]), "hyp_i": hi}

    keys = list(chosen)
    overlap = set(rng.sample(range(len(keys)), n_overlap))
    items = []
    for i in sorted(range(len(keys)), key=lambda i: (i not in overlap, i)):
        k, stratum = keys[i], chosen[keys[i]]
        main = stratum != "other_models"
        items.append({"row": None, "target": 3 if i in overlap else 1, "block": 0,
                      "payload": {"ref_word": k[0], "hyp_word": k[1],
                                  "examples": [context(*x) for x in (main_ex if main else oth_ex)[k]]},
                      "meta": {"stratum": stratum, "model": MAIN if main else "other_models",
                               "rule_type": (main_t if main else oth_t)[k], "count": (main_c if main else oth_c)[k]}})
    info = {"main_substitutions": sum(main_c.values()),
            "top_share_of_main_substitutions": sum(main_c[k] for k in ranked[:n_top]) / max(1, sum(main_c.values()))}
    return items, info


class Supabase:
    """Minimal REST client (service key): upsert rows, upload private storage objects."""

    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.h = {"apikey": key, "Authorization": f"Bearer {key}"}

    def upsert(self, table, rows):
        r = requests.post(f"{self.url}/rest/v1/{table}", json=rows, timeout=60,
                          headers={**self.h, "Prefer": "resolution=merge-duplicates,return=minimal"})
        r.raise_for_status()

    def upload(self, bucket, path, data, content_type):
        r = requests.post(f"{self.url}/storage/v1/object/{bucket}/{path}", data=data, timeout=60,
                          headers={**self.h, "Content-Type": content_type, "x-upsert": "true"})
        r.raise_for_status()

    def count(self, table):
        r = requests.get(f"{self.url}/rest/v1/{table}?select=id", timeout=60,
                         headers={**self.h, "Prefer": "count=exact", "Range": "0-0"})
        r.raise_for_status()
        return int(r.headers["Content-Range"].split("/")[-1])


def read_env(path=".env"):
    env = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=499)
    p.add_argument("--dry_run", action="store_true", help="build items and clips locally, upload nothing")
    args = p.parse_args()

    refs = ref_items(args.seed)
    audits, audit_info = audit_items({it["meta"]["utt_id"] for it in refs}, args.seed)
    pairs, pair_info = pair_items(args.seed)
    rows = []
    for prefix, task, items in [("R", "ref", refs), ("A", "audit", audits), ("P", "pair", pairs)]:
        for k, it in enumerate(items):
            # Priority: block order, agreement items first inside a block, then sampling order.
            rows.append({"id": f"{prefix}{k + 1:04d}", "task": task,
                         "priority": it["block"] * 10000 + (0 if it["target"] > 1 else 5000) + k,
                         "target": it["target"], "clip": None, "payload": it["payload"], "meta": it["meta"],
                         "_row": it["row"]})

    clip_dir = CACHE / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for r in tqdm([r for r in rows if r["_row"] is not None], desc="clips", mininterval=5):
        r["clip"] = f"{r['id']}.mp3"
        f = clip_dir / r["clip"]
        if not f.exists():
            sf.write(f, load_row_audio(r["_row"]), 16000, format="MP3")

    summary = {"seed": args.seed, "counts": Counter(r["task"] for r in rows),
               "labels_needed": {t: sum(r["target"] for r in rows if r["task"] == t) for t in ("ref", "audit", "pair")},
               "ref_sources": Counter(r["meta"]["source"] for r in rows if r["task"] == "ref"),
               "audit": audit_info, "pairs": pair_info,
               "audit_units": sum(len(r["payload"]["units"]) for r in rows if r["task"] == "audit"),
               "clips_mb": round(sum((clip_dir / r["clip"]).stat().st_size for r in rows if r["clip"]) / 1e6, 1)}
    manifest = [{k: v for k, v in r.items() if k != "_row"} for r in rows]
    atomic_write_text(CACHE / "items.json", json.dumps(manifest, ensure_ascii=False))
    atomic_write_text(CACHE / "summary.json", json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    if args.dry_run:
        return

    env = read_env()
    if not env.get("SUPABASE_URL") or not env.get("SUPABASE_SECRET_KEY"):
        raise SystemExit("Put SUPABASE_URL and SUPABASE_SECRET_KEY in .env first")
    sb = Supabase(env["SUPABASE_URL"], env["SUPABASE_SECRET_KEY"])
    with RunRecord(CACHE / "run_info.json", inputs=["data/splits_fa/eval.csv"]) as rec:
        done_path = CACHE / "uploaded.txt"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        with open(done_path, "a") as log:
            for r in tqdm([r for r in rows if r["clip"] and r["clip"] not in done], desc="upload clips", mininterval=5):
                sb.upload("clips", r["clip"], (clip_dir / r["clip"]).read_bytes(), "audio/mpeg")
                log.write(r["clip"] + "\n")
                log.flush()
        for i in tqdm(range(0, len(manifest), 200), desc="upload items"):
            sb.upsert("items", manifest[i:i + 200])
        print(f"items in Supabase: {sb.count('items')}")
        rec.outputs = [CACHE / "items.json", CACHE / "summary.json"]


if __name__ == "__main__":
    main()
