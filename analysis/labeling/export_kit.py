"""
Build the team labeling kits (Phase 2, E5/E6): one self-contained folder per
annotator, opened by double-clicking index.html (works offline, no install).

Task 1, verified references ("verified-300"): 300 FA eval chunks, sampled at
random, stratified by recording (proportional to chunk count). The annotator
listens and corrects the reference. 30 chunks are also given to a second
annotator to measure agreement. No model output is shown (the evaluated model
must not influence the references).

Task 2, word pairs ("top-500"): substitution pairs (reference word -> output
word) from the main model's greedy errors on all test sets: the 300 most
frequent pairs plus, for each rule type, the 50 most frequent pairs not yet
chosen (so every rule type can be checked). 100 random pairs go to every
annotator (agreement), the rest are split. The rule type is not shown.

Output: labeling_kit/<annotator>/{index.html, data.js, clips/*.mp3} and
labeling_kit/manifest.json (what was sampled, seeds, sources). Labels are
saved in the browser and downloaded as CSV; analysis/labeling/import_labels.py
reads them back.

Usage:
  PYTHONPATH=src ./Scripts/python.exe analysis/labeling/export_kit.py --annotators Omi,Nafisa,Third
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, "src")
from bangla_asr.common.audio import load_segment  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import read_jsonl  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

HERE = Path(__file__).parent
MODEL = "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"
SETS = ["splits_fa-eval", "fleurs-test", "kathbath-valid"]
PAIR_TYPES = ["convention", "vowel_shift", "near_miss", "substitution"]


def sample_refs(df, n, seed):
    """n chunks, stratified by recording with largest-remainder allocation."""
    sizes = df.groupby("recording").size()
    quota = sizes / sizes.sum() * n
    alloc = quota.astype(int)
    for rec in (quota - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[rec] += 1
    parts = [df[df["recording"] == r].sample(k, random_state=seed) for r, k in alloc.items() if k]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def select_pairs(n_top, n_per_type, n_examples, seed):
    counts, types, examples = Counter(), {}, defaultdict(list)
    rng = random.Random(seed)
    for s in SETS:
        err_path = Path("results/error_types", s, MODEL, "errors.csv")
        preds = {r["utt_id"]: r for r in read_jsonl(Path("results/zeroshot", s, MODEL, "predictions.jsonl"))}
        e = pd.read_csv(err_path, encoding="utf-8-sig", dtype={"ref_word": str, "hyp_word": str})
        e = e[(e["op"] == "substitute") & (e["type"] != "segmentation")]
        for r in e.itertuples():
            key = (r.ref_word, r.hyp_word)
            counts[key] += 1
            types[key] = r.type
            ex = examples[key]
            # Reservoir sample of n_examples contexts per pair.
            item = (s, r.utt_id, int(r.ref_index), int(r.hyp_index))
            if len(ex) < n_examples:
                ex.append(item)
            elif rng.random() < n_examples / counts[key]:
                ex[rng.randrange(n_examples)] = item
    ranked = [k for k, _ in counts.most_common()]
    chosen = {k: "top" for k in ranked[:n_top]}
    for t in PAIR_TYPES:
        extra = [k for k in ranked if k not in chosen and types[k] == t][:n_per_type]
        chosen.update({k: f"type:{t}" for k in extra})

    def context(s, utt, ri, hi):
        r = PREDS_CACHE.setdefault(s, {x["utt_id"]: x for x in
                                       read_jsonl(Path("results/zeroshot", s, MODEL, "predictions.jsonl"))})[utt]
        return {"set": s, "ref": normalize_bangla_text(r["ref"]).split(), "ref_i": ri,
                "hyp": normalize_bangla_text(r["hyp"]).split(), "hyp_i": hi}

    rows = []
    for i, (k, stratum) in enumerate(chosen.items()):
        rows.append({"id": f"P{i + 1:03d}", "ref_word": k[0], "hyp_word": k[1], "count": counts[k],
                     "rule_type": types[k], "stratum": stratum,
                     "examples": [context(*x) for x in examples[k]]})
    return rows, sum(counts.values())


PREDS_CACHE = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotators", default="A,B,C", help="comma-separated names, one kit each")
    p.add_argument("--n_refs", type=int, default=300)
    p.add_argument("--ref_overlap", type=int, default=30)
    p.add_argument("--n_top_pairs", type=int, default=300)
    p.add_argument("--n_pairs_per_type", type=int, default=50)
    p.add_argument("--pair_overlap", type=int, default=100)
    p.add_argument("--seed", type=int, default=499)
    p.add_argument("--out", default="labeling_kit")
    args = p.parse_args()

    names = [a.strip() for a in args.annotators.split(",") if a.strip()]
    out = Path(args.out)
    with RunRecord(out / "run_info.json", inputs=["data/splits_fa/eval.csv"]) as rec:
        # Task 1: references
        eval_df = pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig")
        refs = sample_refs(eval_df, args.n_refs, args.seed)
        refs["id"] = [f"R{i + 1:03d}" for i in range(len(refs))]
        share = [refs.iloc[k::len(names)] for k in range(len(names))]
        overlap_per = args.ref_overlap // len(names)
        ref_assign = {n: list(share[k]["id"]) for k, n in enumerate(names)}
        for k, n in enumerate(names):  # each annotator also does a few of the previous one's chunks
            prev = names[k - 1]
            ref_assign[n] += list(share[(k - 1) % len(names)]["id"][:overlap_per]) if prev != n else []

        # Task 2: word pairs
        pairs, n_sub = select_pairs(args.n_top_pairs, args.n_pairs_per_type, 2, args.seed)
        rng = random.Random(args.seed)
        ids = [x["id"] for x in pairs]
        common = set(rng.sample(ids, min(args.pair_overlap, len(ids))))
        rest = [i for i in ids if i not in common]
        pair_assign = {n: sorted(common) + rest[k::len(names)] for k, n in enumerate(names)}

        template = (HERE / "app.html").read_text(encoding="utf-8")
        ref_by_id = refs.set_index("id")
        pair_by_id = {x["id"]: x for x in pairs}
        for n in names:
            kit = out / n
            (kit / "clips").mkdir(parents=True, exist_ok=True)
            items = []
            for i in ref_assign[n]:
                r = ref_by_id.loc[i]
                clip = kit / "clips" / f"{i}.mp3"
                if not clip.exists():
                    sf.write(clip, load_segment(r["audio_path"], r["start"], r["end"]), 16000, format="MP3")
                items.append({"id": i, "utt_id": r["utt_id"], "clip": f"clips/{i}.mp3",
                              "duration": round(float(r["end"] - r["start"]), 2), "text": str(r["text"])})
            data = {"kit": f"phase2-{args.seed}", "annotator": n, "refs": items,
                    "pairs": [{k: v for k, v in pair_by_id[i].items() if k not in ("rule_type", "stratum", "count")}
                              for i in pair_assign[n]]}
            (kit / "data.js").write_text("window.KIT = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                         encoding="utf-8")
            (kit / "index.html").write_text(template, encoding="utf-8")
            print(f"{n}: {len(items)} chunks, {len(data['pairs'])} pairs -> {kit}")

        manifest = {
            "seed": args.seed, "annotators": names, "model_for_pairs": MODEL, "sets_for_pairs": SETS,
            "substitutions_considered": n_sub,
            "refs": refs[["id", "utt_id", "recording", "start", "end", "text"]].to_dict("records"),
            "ref_assignment": ref_assign,
            "pairs": [{k: v for k, v in x.items() if k != "examples"} for x in pairs],
            "pair_assignment": pair_assign, "pair_overlap_ids": sorted(common),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        rec.outputs = [out / "manifest.json"]
    top_mass = sum(x["count"] for x in pairs if x["stratum"] == "top")
    print(f"pairs: {len(pairs)} chosen; the top {args.n_top_pairs} cover {100 * top_mass / n_sub:.1f}% "
          f"of {n_sub} substitutions")


if __name__ == "__main__":
    main()
