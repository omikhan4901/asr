#!/usr/bin/env python3
"""
Phase 4 inference on Kaggle (T4 x2): everything comes from the private Hugging Face repo HUB
(audio zips, manifests, models, A100 reference scores); the outputs go to /kaggle/working.
Each GPU runs its own queue of jobs; both run at the same time.

Jobs
  t4check        zero-shot tugstugi, greedy batch 8, on dev / FLEURS / Kathbath (-> tugstugi_t4); compared
                 per utterance with the same model on the Modal A100 (tugstugi_a100). Kaggle rows count only
                 if these match.
  lora           M1 LoRA (models/m1_lora on tugstugi), greedy batch 8, on Kathbath (all) and Common Voice
                 (-> m1_lora_t4); the Kathbath clips the A100 already did are the per-utterance check.
  stack:<set>    stacking input for the best M1 (models/m1_full) on dev | eval | fleurs | kathbath:
                 sampled 12-best (T 0.7, as in Phases 1-3) -> exact scores (teacher forcing) -> pool + greedy
                 (the A100 greedy of m1_full), i.e. the same pool format as data/phase3/*_pool.jsonl.
  score:<run>    any model in models/<run> (full model or LoRA adapter) on the 5 test sets (matrix runs).

Every step is resumable (the scripts skip finished utterances), and outputs are written straight to
/kaggle/working, so a second session continues where the first stopped (add the first session's
output as an input dataset and pass --resume_from).

  python pipeline/kaggle_phase4.py --gpu0 t4check --gpu1 lora
  python pipeline/kaggle_phase4.py --gpu0 stack:eval --gpu1 stack:dev,stack:fleurs,stack:kathbath
Local test (CPU, two utterances, a local mirror of the hub):
  python pipeline/kaggle_phase4.py --gpu0 t4check --hub_dir <mirror> --work <tmp> --out <tmp> --cpu --limit 2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HUB = "OmiKhan/bangla-asr-hub"
TUGSTUGI = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
REPO = Path(__file__).resolve().parent.parent
# set name -> manifest inside the work dir (same paths as on Modal, so result folders get the same names)
SETS = {"splits_fa-dev": "data/splits_fa/dev.csv", "splits_fa-eval": "data/splits_fa/eval.csv",
        "fleurs-test": "data/external/fleurs/test.csv", "kathbath-valid": "data/external/kathbath/valid.csv",
        "cv-test": "data/external/cv/test.csv"}
SHORT = {"dev": "splits_fa-dev", "eval": "splits_fa-eval", "fleurs": "fleurs-test", "kathbath": "kathbath-valid"}
TEST_SETS = list(SETS)


def job_needs(job: str) -> tuple[list[str], list[str]]:
    """(sets whose audio is needed, models needed) for one job."""
    kind, _, arg = job.partition(":")
    if kind == "t4check":
        return ["splits_fa-dev", "fleurs-test", "kathbath-valid"], []
    if kind == "lora":
        return ["kathbath-valid", "cv-test"], ["m1_lora"]
    if kind == "stack":
        return [SHORT[arg]], ["m1_full"]
    if kind == "score":
        return TEST_SETS, [arg]
    raise SystemExit(f"unknown job {job!r}")


def job_commands(job: str, py: str, limit: list[str]) -> list[list[str]]:
    kind, _, arg = job.partition(":")
    ev = [py, str(REPO / "pipeline/eval_zero_shot.py"), "--batch_size", "8", *limit]
    if kind == "t4check":
        return [ev + ["--manifest", SETS[s], "--model", TUGSTUGI, "--name", "tugstugi_t4", "--out_dir", "results/zeroshot"]
                for s in job_needs(job)[0]]
    if kind == "lora":
        return [ev + ["--manifest", SETS[s], "--model", TUGSTUGI, "--adapter", "models/m1_lora", "--name", "m1_lora_t4",
                      "--out_dir", "results/finetuned"] for s in ("kathbath-valid", "cv-test")]
    if kind == "score":
        model = ["--model", TUGSTUGI, "--adapter", f"models/{arg}"] if Path(f"models/{arg}/adapter_config.json").exists() \
            else ["--model", f"models/{arg}"]
        return [ev + ["--manifest", SETS[s], *model, "--name", f"{arg}_t4", "--out_dir", "results/finetuned"] for s in TEST_SETS]
    s = SHORT[arg]  # stack
    nb, man = f"nbest/{s}__m1_full", SETS[s]
    return [[py, str(REPO / "pipeline/03_generate_nbest.py"), "--split_csv", man, "--model_id", "models/m1_full",
             "--output_jsonl", f"{nb}.jsonl", "--decoding", "sampled", "--num_beams", "12", "--temperature", "0.7",
             "--batch_size", "2", *limit],
            [py, str(REPO / "pipeline/rescore_nbest.py"), "--input_jsonl", f"{nb}.jsonl", "--manifest", man,
             "--model_id", "models/m1_full", "--output_jsonl", f"{nb}_scored.jsonl"],
            [py, str(REPO / "pipeline/rescore_nbest.py"), "--input_jsonl", f"{nb}_scored.jsonl", "--manifest", man,
             "--model_id", "models/m1_full", "--output_jsonl", f"{nb}_pool.jsonl",
             "--add_greedy", f"ref/finetuned/{s}/m1_full/predictions.jsonl"]]


def fetch(hub_dir: Path, sets: set, models: set, base: bool, token: str | None):
    """Download only what the jobs need from HUB into hub_dir; the base model (tugstugi) once into the HF cache,
    so the two GPU queues do not download the same 3 GB file at the same time."""
    from huggingface_hub import snapshot_download

    zips = {"cv-test": "zips/cv-test.zip"}
    pats = ["manifests/*", "results/*"] + [zips.get(s, f"zips/{s}.zip") for s in sets] + [f"models/{m}/*" for m in models]
    t0 = time.time()
    snapshot_download(HUB, repo_type="dataset", allow_patterns=pats, local_dir=str(hub_dir), token=token)
    print(f"downloaded {pats} in {time.time() - t0:.0f} s", flush=True)
    if base:
        t0 = time.time()
        snapshot_download(TUGSTUGI, allow_patterns=["model.safetensors", "*config.json", "tokenizer.json", "vocab.json",
                                                    "merges.txt", "added_tokens.json", "special_tokens_map.json",
                                                    "normalizer.json"])
        print(f"downloaded {TUGSTUGI} in {time.time() - t0:.0f} s", flush=True)


def layout(hub_dir: Path, work: Path, out: Path, sets: set, models: set, limit_rows: int | None):
    """Work dir like the Modal container: data/clips/<set>/..., data/splits_fa = manifests,
    data/external = manifests/external, models/, ref/ = A100 scores; results/ and nbest/ live in out/."""
    clips = work / "data" / "clips"
    for s in sorted(sets):
        folder = clips / ("cv-test-mp3" if s == "cv-test" else s)
        if not folder.exists():
            with zipfile.ZipFile(hub_dir / "zips" / f"{s}.zip") as z:
                z.extractall(clips)
            print(f"unpacked {s}: {sum(1 for _ in folder.rglob('*.*'))} files", flush=True)
    links = {work / "data" / "splits_fa": hub_dir / "manifests", work / "data" / "external": hub_dir / "manifests" / "external",
             work / "models": hub_dir / "models", work / "ref": hub_dir / "results",
             work / "results": out / "results", work / "nbest": out / "nbest"}
    for link, target in links.items():
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError:  # Windows test run without symlink rights: a directory junction
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    for m in models:
        assert (hub_dir / "models" / m).exists(), f"models/{m} is not in the hub: run to_hf on Modal first"


def drop_old_torchao():
    """Kaggle's image ships torchao 0.10.0; peft 0.21.2 refuses to load any LoRA adapter while a torchao older
    than 0.16 is installed (ImportError in is_torchao_available). Nothing here uses torchao (Modal has none)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("torchao")
    except PackageNotFoundError:
        return
    if tuple(int(x) for x in v.split(".")[:2] if x.isdigit()) < (0, 16):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
        print(f"removed torchao {v} (too old for peft, not used)", flush=True)


def run_queue(gpu: int, jobs: list[str], args, logs: Path, status: dict):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1" if args.cpu else str(gpu), "PYTHONPATH": str(REPO / "src"),
           "PYTHONIOENCODING": "utf-8", "TQDM_MININTERVAL": "60"}
    limit = ["--limit", str(args.limit)] if args.limit else []
    for job in jobs:
        status[job] = "running"
        log = logs / f"{job.replace(':', '_')}.log"
        t0 = time.time()
        for cmd in job_commands(job, sys.executable, limit):
            with log.open("a", encoding="utf-8") as f:
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                code = subprocess.run(cmd, cwd=args.work, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if code:
                status[job] = f"FAILED (exit {code}, see {log.name})"
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                print(f"[GPU {gpu}] {job}: last lines of {log.name}:\n" + "\n".join(tail), flush=True)
                break
        else:
            status[job] = f"done in {(time.time() - t0) / 60:.0f} min"
        print(f"[GPU {gpu}] {job}: {status[job]}", flush=True)


def compare(out: Path, hub_dir: Path) -> dict:
    """Per-utterance check of the T4 rows against the A100 rows of the same model."""
    pairs = [("zeroshot", "tugstugi_t4", "tugstugi_a100"), ("finetuned", "m1_lora_t4", "m1_lora")]
    res = {}
    for kind, t4, a100 in pairs:
        for p in sorted((out / "results" / kind).glob(f"*/{t4}/predictions.jsonl")):
            s = p.parts[-3]
            q = hub_dir / "results" / kind / s / a100 / "predictions.jsonl"
            if not q.exists():
                continue
            A = {json.loads(l)["utt_id"]: json.loads(l) for l in p.open(encoding="utf-8")}
            B = {json.loads(l)["utt_id"]: json.loads(l) for l in q.open(encoding="utf-8")}
            common = sorted(A.keys() & B.keys())
            if not common:
                continue
            same = sum(A[u]["hyp"] == B[u]["hyp"] for u in common)
            res[f"{s}/{t4}_vs_{a100}"] = {"utterances": len(common), "identical": same,
                                         "identical_share": round(same / len(common), 4)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu0", default="")
    ap.add_argument("--gpu1", default="")
    ap.add_argument("--hub_dir", default="/tmp/hub")
    ap.add_argument("--work", default="/tmp/work")
    ap.add_argument("--out", default="/kaggle/working/phase4")
    ap.add_argument("--resume_from", default=None, help="an earlier session's phase4/ folder (added as input)")
    ap.add_argument("--no_fetch", action="store_true", help="hub_dir is already a full local mirror (tests)")
    ap.add_argument("--cpu", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=None, help="tests only: N utterances per set")
    args = ap.parse_args()
    queues = [[j for j in q.split(",") if j] for q in (args.gpu0, args.gpu1)]
    jobs = [j for q in queues for j in q]
    if not jobs:
        raise SystemExit("no jobs: pass --gpu0 and/or --gpu1")
    sets, models = set(), set()
    for j in jobs:
        s, m = job_needs(j)
        sets |= set(s)
        models |= set(m)
    hub_dir, work, out = Path(args.hub_dir), Path(args.work), Path(args.out)
    for d in (hub_dir, work, out / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    if args.resume_from and not (out / "results").exists():
        shutil.copytree(args.resume_from, out, dirs_exist_ok=True)
        print(f"continuing from {args.resume_from}", flush=True)
    if not args.no_fetch:
        base = any(j.split(":")[0] in ("t4check", "lora") for j in jobs)
        fetch(hub_dir, sets, models, base, os.environ.get("HF_TOKEN"))
    layout(hub_dir, work, out, sets, models, args.limit)
    drop_old_torchao()

    status, threads = {}, []
    for gpu, q in enumerate(queues):
        if q:
            t = threading.Thread(target=run_queue, args=(gpu, q, args, out / "logs", status))
            t.start()
            threads.append(t)
    while any(t.is_alive() for t in threads):
        time.sleep(300)
        print(time.strftime("%H:%M"), json.dumps(status), flush=True)
    for t in threads:
        t.join()

    summary = {"jobs": status, "gpu_check": compare(out, hub_dir), "time": time.strftime("%Y-%m-%d %H:%M")}
    for m in (out / "results").rglob("metrics.json"):
        summary.setdefault("metrics", {})[str(m.relative_to(out))] = json.loads(m.read_text(encoding="utf-8")).get("wer")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    shutil.make_archive(str(out.parent / "phase4_outputs"), "zip", out.parent, out.name)
    print(f"ready to download: {out.parent / 'phase4_outputs.zip'}")
    if any(not v.startswith("done") for v in status.values()):
        raise SystemExit("some jobs did not finish: see the logs folder; run the session again to continue")


if __name__ == "__main__":
    main()
