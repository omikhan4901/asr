"""
Cross-model ROVER (Phase 5 idea 2): word-level voting over the greedy outputs of DIFFERENT fine-tuned models
(different training data -> different errors), unlike Phase 3/4 ROVER over one model's 12-best.

Members are chosen on DL Sprint dev only (pre-registered rule): start from the best single model on dev as the
backbone, add the model that lowers dev WER most, stop when nothing helps. A second, unselected ensemble uses
every model. Both are applied unchanged to the test sets and compared with M1 full greedy (paired bootstrap).
Not a deployable system (N models = N x the decoding time): it is the candidate teacher for pseudo-labels.

Input:  results/modal/{finetuned,zeroshot}/<set>/<run>/predictions.jsonl, results/kaggle/finetuned/<set>/<run>_t4/
Output: results/ensemble/<set>/<name>/predictions.jsonl, results/ensemble/summary.json, paired tests in
        results/compare/ensemble/
Run:    PYTHONPATH=src ./Scripts/python.exe analysis/cross_model_rover.py
"""

import json
import subprocess
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.rescoring.combination import backbone_votes, combine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = ["splits_fa-dev", "splits_fa-eval", "fleurs-test", "cv-test", "kathbath-valid"]
RUNS = ["m1_full", "m1_lora", "m1_decoder", "m1_encoder", "m1_full_4ep", "m2_cv", "m3_kathbath", "m4_shrutilipi",
        "m5_merged", "m6_harmonized", "tugstugi_a100"]


def load(p):
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    return {r["utt_id"]: r for r in rows}


def find(s, run, n):
    for p in [ROOT / "results/modal/finetuned" / s / run, ROOT / "results/modal/zeroshot" / s / run,
              ROOT / "results/kaggle/finetuned" / s / f"{run}_t4"]:
        f = p / "predictions.jsonl"
        if f.exists() and sum(1 for _ in f.open(encoding="utf-8")) == n:
            return f
    return None


def toks(t):
    return normalize_bangla_text(t or "").split()


def ensemble(members, P, ref, utts):
    """members[0] is the backbone; one vote per member."""
    out, edits, words = {}, 0, 0
    for u in utts:
        b = P[members[0]][u]
        votes = [backbone_votes(b, P[m][u]) for m in members]
        hyp = combine(len(b), votes, [1.0] * len(members)).split()
        out[u] = hyp
        edits += Levenshtein.distance(ref[u], hyp)
        words += len(ref[u])
    return out, edits / max(1, words)


def wer(run, P, ref, utts):
    return sum(Levenshtein.distance(ref[u], P[run][u]) for u in utts) / sum(len(ref[u]) for u in utts)


def main():
    n_ref = {s: sum(1 for _ in (ROOT / "results/modal/zeroshot" / s / "tugstugi_a100/predictions.jsonl").open(encoding="utf-8"))
             for s in SETS}
    paths = {s: {r: find(s, r, n_ref[s]) for r in RUNS} for s in SETS}
    runs = [r for r in RUNS if all(paths[s][r] for s in SETS)]
    print("models complete on all 5 sets:", runs)
    data = {}
    for s in SETS:
        raw = {r: load(paths[s][r]) for r in runs}
        utts = sorted(raw[runs[0]])
        ref = {u: toks(raw[runs[0]][u]["ref"]) for u in utts}
        P = {r: {u: toks(raw[r][u]["hyp"]) for u in utts} for r in runs}
        data[s] = (P, ref, utts, raw[runs[0]])

    P, ref, utts, _ = data["splits_fa-dev"]
    singles = {r: wer(r, P, ref, utts) for r in runs}
    chosen = [min(singles, key=singles.get)]
    best = singles[chosen[0]]
    log = [{"members": list(chosen), "dev_wer": round(100 * best, 2)}]
    while True:
        trials = {r: ensemble(chosen + [r], P, ref, utts)[1] for r in runs if r not in chosen}
        if not trials:
            break
        r = min(trials, key=trials.get)
        if trials[r] >= best:
            break
        chosen.append(r)
        best = trials[r]
        log.append({"members": list(chosen), "dev_wer": round(100 * best, 2)})
    print("dev selection:", json.dumps(log, ensure_ascii=False))
    systems = {"rover_dev_selected": chosen, "rover_all": [chosen[0]] + [r for r in runs if r != chosen[0]]}

    summary = {"dev_singles": {r: round(100 * w, 2) for r, w in singles.items()}, "selection": log, "systems": systems,
               "results": {}}
    for s in SETS:
        P, ref, utts, base = data[s]
        for name, members in systems.items():
            hyp, w = ensemble(members, P, ref, utts)
            out = ROOT / "results/ensemble" / s / name
            out.mkdir(parents=True, exist_ok=True)
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
                for u in utts:
                    f.write(json.dumps({"utt_id": u, "group": base[u].get("group", u), "ref": base[u]["ref"],
                                        "hyp": " ".join(hyp[u])}, ensure_ascii=False) + "\n")
            summary["results"].setdefault(s, {})[name] = round(100 * w, 2)
            if s != "splits_fa-dev":
                subprocess.run([sys.executable, str(ROOT / "analysis/paired_compare.py"), "--a", str(paths[s]["m1_full"]),
                                "--b", str(out / "predictions.jsonl"), "--name", f"{s}__{name}_vs_m1_full",
                                "--out_dir", str(ROOT / "results/compare/ensemble")], check=True)
        summary["results"][s]["m1_full (greedy)"] = round(100 * wer("m1_full", P, ref, utts), 2)
    (ROOT / "results/ensemble/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
