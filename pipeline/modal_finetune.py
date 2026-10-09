#!/usr/bin/env python3
"""
Phase 4 on Modal: fine-tune with pipeline/11_finetune.py and score with pipeline/eval_zero_shot.py.

Both scripts are the same files as on the RTX 3060, called with the same arguments and the
same manifest paths (so config.json and the results' set names match the local runs). Only
the audio source differs:
  DL Sprint train/dev/eval, FLEURS, Kathbath  clips exported by pipeline/export_kaggle.py
      (16 kHz FLAC; checked bit-identical to the audio load_row_audio reads locally)
  Common Voice  the original mp3 files (resampled in the container exactly as locally;
      a 16-bit export of resampled audio would not be bit-identical)
The Modal manifests are copies of the local ones with audio_path pointing at those files and
start/end emptied; every other column is unchanged.

Volume "bangla-asr-ft":
  /zips/<set>.zip            audio (uploaded once)
  /manifests/...             manifests, laid out so data/splits_fa/* and data/external/* resolve
  /checkpoints/<run>/        fine-tuning runs (resumable), uploaded local runs (LoRA adapter)
  /results/<out>/<set>/<run>/  predictions.jsonl, metrics.json, run_info.json (resumable)
  /hf/                       Hugging Face cache

Commands (Git Bash, repo root; prefix every command with  PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run):
  pipeline/modal_finetune.py::upload                       audio + manifests (skips zips already there)
  --detach pipeline/modal_finetune.py::smoke --method full   ~$1 test of the training path
  --detach pipeline/modal_finetune.py::train --method full --run m1_full --budget 5
  pipeline/modal_finetune.py::pull --run m1_full            (--logs-only: no weights)
  pipeline/modal_finetune.py::upload_run --local checkpoints/m1_lora --run m1_lora   (final/ + logs)
  --detach pipeline/modal_finetune.py::score --run m1_full --model checkpoints/m1_full/final --budget 3
  --detach pipeline/modal_finetune.py::score --run m1_lora --adapter checkpoints/m1_lora/final --budget 3
  pipeline/modal_finetune.py::pull_results --run m1_full    -> results/modal/finetuned/<set>/m1_full/
--detach keeps a job running when this terminal closes or the internet drops.

Money safety: --budget caps a whole run in dollars (all calls and restarts; spend is tracked on the
volume at the price below). Before the cap the job gets one Ctrl+C: training finishes its step,
saves and exports partial/ (a scorable model); scoring keeps every finished batch. At the cap the
job is killed; training keeps its last periodic checkpoint (every 100 steps), scoring its finished
batches. Run the same command again with a higher --budget to continue.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import modal

APP = "bangla-asr-finetune"
VOLUME = "bangla-asr-ft"
GPU = "A100-40GB"  # full AdamW state of Whisper-medium (~12 GB) + activations fit; one GPU type for all Phase 4 rows
TUGSTUGI = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
# Cost per hour of one container (modal.com/pricing, 2026-10-10): A100 40 GB $2.10 + 4 physical cores x $0.047
# + 24 GiB x $0.008. Used only for the budget cap; the Modal dashboard shows the billed amount.
RATE_PER_H = 2.10 + 4 * 0.047 + 24 * 0.008
RESERVE_MIN = 15  # minutes kept back for the last dev check, the checkpoint save and the partial/ export
REMOTE_PROJ, WORK = "/root/proj", "/root/work"

# set -> local zip, local manifest, manifest path under /vol/manifests, audio kind.
# /vol/manifests is mounted as data/splits_fa, /vol/manifests/external as data/external, so every
# manifest keeps its local path (eval_zero_shot.py names results after it: splits_fa-eval, fleurs-test, ...).
SETS = {
    "splits_fa-train": ("data/kaggle_hishab/zips/splits_fa-train.zip", "data/splits_fa/train_clean.csv", "train_clean.csv", "flac"),
    "splits_fa-dev": ("data/kaggle_hishab/zips/splits_fa-dev.zip", "data/splits_fa/dev.csv", "dev.csv", "flac"),
    "splits_fa-eval": ("data/kaggle_hishab/zips/splits_fa-eval.zip", "data/splits_fa/eval.csv", "eval.csv", "flac"),
    "fleurs-test": ("data/kaggle_hishab/zips/fleurs-test.zip", "data/external/fleurs/test.csv", "external/fleurs/test.csv", "flac"),
    "kathbath-valid": ("data/kaggle_hishab/zips/kathbath-valid.zip", "data/external/kathbath/valid.csv", "external/kathbath/valid.csv", "flac"),
    "cv-test": ("data/modal/cv-test-mp3.zip", "data/external/cv/test.csv", "external/cv/test.csv", "mp3"),
}
TRAIN_SETS = ["splits_fa-train", "splits_fa-dev"]
TEST_SETS = ["splits_fa-dev", "splits_fa-eval", "fleurs-test", "kathbath-valid", "cv-test"]  # runs/phase4 order

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
# Same library versions as the local venv (run_info.json records them).
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.13.0", "transformers==5.15.0", "peft==0.21.2", "accelerate==1.14.0",
                 "safetensors==0.8.0", "numpy==2.5.2", "pandas==3.0.5", "soundfile==0.14.0",
                 "rapidfuzz==3.14.5", "jiwer==4.0.0", "librosa==1.0.0", "soxr==1.1.0", "tqdm==4.70.0")
    .env({"PYTHONPATH": f"{REMOTE_PROJ}/src", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
          "HF_HOME": "/vol/hf", "TQDM_MININTERVAL": "30"})
    .add_local_dir("src/bangla_asr", f"{REMOTE_PROJ}/src/bangla_asr", ignore=["**/__pycache__"])
    .add_local_file("pipeline/11_finetune.py", f"{REMOTE_PROJ}/pipeline/11_finetune.py")
    .add_local_file("pipeline/checkpoint_to_model.py", f"{REMOTE_PROJ}/pipeline/checkpoint_to_model.py")
    .add_local_file("pipeline/eval_zero_shot.py", f"{REMOTE_PROJ}/pipeline/eval_zero_shot.py")
)


# ---------------------------------------------------------------- local side: data

def _cv_zip(manifest: str, dst: str):
    """Zip the original Common Voice mp3s of the test manifest (stored, mp3 is already compressed)."""
    import pandas as pd

    if Path(dst).exists():
        return
    df = pd.read_csv(manifest, encoding="utf-8-sig")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(dst + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in df["audio_path"]:
            z.write(p, f"cv-test-mp3/clips/{Path(p).name}")
    tmp.replace(dst)
    print(f"built {dst} ({Path(dst).stat().st_size / 1e9:.2f} GB, {len(df)} mp3)")


def build_manifest(set_name: str, out: Path) -> Path:
    """Modal copy of one manifest: audio_path -> the uploaded file, start/end emptied; checked against the zip."""
    import pandas as pd

    zpath, src, _, kind = SETS[set_name]
    df = pd.read_csv(src, encoding="utf-8-sig")
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("." + kind)]
    if kind == "flac":
        have, want = {Path(n).stem for n in names}, list(df["utt_id"])
        df["audio_path"] = [f"data/clips/{set_name}/audio/{u}.flac" for u in df["utt_id"]]
    else:
        have, want = {Path(n).name for n in names}, [Path(p).name for p in df["audio_path"]]
        df["audio_path"] = [f"data/clips/cv-test-mp3/clips/{Path(p).name}" for p in df["audio_path"]]
    missing = set(want) - have
    if missing or len(have) != len(df):
        sys.exit(f"{zpath} does not match {src}: {len(missing)} files missing, {len(have)} in zip vs {len(df)} rows")
    for col in ("start", "end"):  # whole clip: load_row_audio falls back to load_audio
        if col in df.columns:
            df[col] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{set_name}: {len(df)} rows ({kind})")
    return out


def _exists(path):
    try:
        vol.listdir(path)
        return True
    except Exception:
        return False


@app.local_entrypoint()
def upload():
    """Upload every set's audio (zips already on the volume with the same size are skipped) and all manifests."""
    import tempfile

    _cv_zip(SETS["cv-test"][1], SETS["cv-test"][0])
    have = {e.path.strip("/"): e.size for e in vol.listdir("/zips")} if _exists("/zips") else {}
    with tempfile.TemporaryDirectory() as tmp:
        manifests = {s: build_manifest(s, Path(tmp) / SETS[s][2]) for s in SETS}
        with vol.batch_upload(force=True) as batch:
            for s, m in manifests.items():
                batch.put_file(str(m), f"/manifests/{SETS[s][2]}")
            for s, (local, *_) in SETS.items():
                size = Path(local).stat().st_size
                if have.get(f"zips/{s}.zip") == size:
                    print(f"{s}.zip already on the volume, skipped")
                    continue
                print(f"uploading {local} ({size / 1e9:.2f} GB); this can take a while")
                batch.put_file(local, f"/zips/{s}.zip")
    print("done:", [(e.path, round(e.size / 1e9, 2)) for e in vol.listdir("/zips")])


@app.local_entrypoint()
def upload_run(local: str, run: str):
    """Upload a locally trained run (final/ + logs, not the resumable checkpoint) to /checkpoints/<run>/ for scoring."""
    root = Path(local)
    if not (root / "final").exists():
        sys.exit(f"{root}/final not found: the run has not finished (use pipeline/checkpoint_to_model.py for partial/)")
    files = [p for p in root.rglob("*") if p.is_file()
             and p.relative_to(root).parts[0] not in ("checkpoint", "checkpoint.old", "checkpoint.tmp")]
    with vol.batch_upload(force=True) as batch:
        for p in files:
            batch.put_file(str(p), f"/checkpoints/{run}/{p.relative_to(root).as_posix()}")
    print(f"uploaded {len(files)} files ({sum(p.stat().st_size for p in files) / 1e6:.0f} MB) -> /checkpoints/{run}/")


# ---------------------------------------------------------------- remote side

def _prepare(sets: list[str]):
    """Unpack the needed audio to local disk (fast reads) and lay out the working dir like the repo."""
    import os

    work, clips = Path(WORK), Path("/tmp/clips")
    (work / "data").mkdir(parents=True, exist_ok=True)
    for s in sets:
        folder = clips / ("cv-test-mp3" if s == "cv-test" else s)
        if not folder.exists():
            with zipfile.ZipFile(f"/vol/zips/{s}.zip") as z:
                z.extractall(clips)
            print(f"unpacked {s}: {sum(1 for _ in folder.rglob('*.*'))} files")
    for link, target in [(work / "data" / "clips", clips), (work / "data" / "splits_fa", Path("/vol/manifests")),
                         (work / "data" / "external", Path("/vol/manifests/external")),
                         (work / "checkpoints", Path("/vol/checkpoints")), (work / "results", Path("/vol/results"))]:
        Path(target).mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(target, link)


def _supervise(cmds: list[list[str]], spend_file: Path, budget: float, launch: dict | None, label: str) -> tuple[int, bool]:
    """Run commands one after another under a dollar cap for the whole run (earlier calls count):
    one Ctrl+C before the cap (clean stop), kill at the cap. Returns (last exit code, hard stopped)."""
    import json
    import signal
    import threading
    import time

    before = json.loads(spend_file.read_text())["dollars"] if spend_file.exists() else 0.0
    t0 = time.time()
    spent = lambda: before + (time.time() - t0) / 3600 * RATE_PER_H  # noqa: E731
    soft = budget - RESERVE_MIN / 60 * RATE_PER_H
    if spent() >= soft:
        sys.exit(f"{label}: ${before:.2f} already spent of a ${budget:.2f} budget (clean stop at ${soft:.2f}). "
                 f"Pass a higher --budget to continue.")
    spend_file.parent.mkdir(parents=True, exist_ok=True)
    if launch:  # the container has no git: record which local commit launched this call
        with spend_file.with_name(spend_file.stem + "_launches.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**launch, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "budget": budget,
                                 "spent_before": round(before, 4), "commands": cmds}) + "\n")
    print(f"{label}: budget ${budget:.2f}, ${before:.2f} spent before; clean stop at ${soft:.2f}, hard stop at "
          f"${budget:.2f} (${RATE_PER_H:.2f}/h for {GPU} + 4 cores + 24 GiB)", flush=True)

    state = {"child": None, "soft": False, "hard": False}

    def stop(*_):
        if not state["soft"]:
            state["soft"] = True
            if state["child"] is not None:
                state["child"].send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)  # Modal preemption interrupts the container: save, then exit
    signal.signal(signal.SIGTERM, stop)

    def watchdog():
        while True:
            spend_file.write_text(json.dumps({"dollars": round(spent(), 4), "rate_per_h": RATE_PER_H}))
            if spent() >= soft and not state["soft"]:
                print(f"[budget] ${spent():.2f} spent: stopping cleanly", flush=True)
                stop()
            if spent() >= budget and not state["hard"]:
                state["hard"] = True
                print(f"[budget] ${spent():.2f} spent: hard stop", flush=True)
                if state["child"] is not None:
                    state["child"].kill()
            time.sleep(10)

    threading.Thread(target=watchdog, daemon=True).start()
    code = 0
    for cmd in cmds:
        if state["soft"]:
            break
        print("$", " ".join(cmd), flush=True)
        # Own session: a signal sent to the whole container must not reach the child twice
        # (a second Ctrl+C makes 11_finetune.py quit without saving); stop() forwards exactly one.
        state["child"] = subprocess.Popen(cmd, cwd=WORK, start_new_session=True)
        code = state["child"].wait()
        vol.commit()
        if code != 0:
            break
    spend_file.write_text(json.dumps({"dollars": round(spent(), 4), "rate_per_h": RATE_PER_H}))
    vol.commit()
    print(f"{label}: exit code {code} | spent about ${spent():.2f} of ${budget:.2f}", flush=True)
    return code, state["hard"]


def _run_finetune(method: str, run: str, extra: list[str], budget: float, launch: dict | None) -> int:
    _prepare(TRAIN_SETS)
    out_dir = f"checkpoints/{run}"
    cmd = [sys.executable, f"{REMOTE_PROJ}/pipeline/11_finetune.py", "--method", method,
           "--train_manifest", "data/splits_fa/train_clean.csv", "--dev_manifest", "data/splits_fa/dev.csv",
           "--out_dir", out_dir, "--num_workers", "4", *extra]
    code, hard = _supervise([cmd], Path("/vol") / out_dir / "modal_spend.json", budget, launch, run)
    if code != 0 and not hard and (Path(WORK) / out_dir / "checkpoint" / "state.pt").exists():
        # Stopped early: also leave a scorable model, so the work done is usable even if nothing runs again.
        subprocess.run([sys.executable, f"{REMOTE_PROJ}/pipeline/checkpoint_to_model.py", "--out_dir", out_dir], cwd=WORK)
        vol.commit()
    print("0 = finished (final/); 3 = stopped early and saved (partial/); run again (higher --budget) to continue")
    return code


GPU_FN = dict(image=image, gpu=GPU, cpu=4, memory=24576, volumes={"/vol": vol})


@app.function(**GPU_FN, timeout=12 * 3600)
def train_remote(method: str, run: str, extra: list[str], budget: float, launch: dict) -> int:
    return _run_finetune(method, run, extra, budget, launch)


@app.function(**GPU_FN, timeout=40 * 60)
def smoke_remote(method: str, launch: dict) -> int:
    # Fresh smoke dir each time; no --max_minutes: the budget watchdog stops it (clean stop at ~$0.88,
    # about 21 min after start), so the smoke test also exercises the stop path and the partial/ export.
    subprocess.run(["rm", "-rf", f"/vol/checkpoints/smoke_{method}"])
    return _run_finetune(method, f"smoke_{method}", ["--dev_limit", "64", "--eval_every", "40", "--memory_probe"],
                         1.5, launch)


@app.function(**GPU_FN, timeout=12 * 3600)
def score_remote(run: str, model: str, adapter: str, sets: list[str], out: str, budget: float, launch: dict) -> int:
    _prepare(sets)
    cmds = []
    for s in sets:
        cmd = [sys.executable, f"{REMOTE_PROJ}/pipeline/eval_zero_shot.py", "--manifest", SETS[s][1],
               "--model", model, "--name", run, "--out_dir", f"results/{out}", "--batch_size", "8"]
        cmds.append(cmd + (["--adapter", adapter] if adapter else []))
    code, _ = _supervise(cmds, Path("/vol/results") / out / "_spend" / f"{run}.json", budget, launch, f"score {run}")
    print("0 = all sets scored; otherwise run again (higher --budget): finished sets and batches are kept")
    return code


# ---------------------------------------------------------------- local entry points

def _git_state() -> dict:
    """Local commit and whether the files shipped to Modal differ from it (run_info.json cannot see git there)."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    shipped = ["src/bangla_asr", "pipeline/11_finetune.py", "pipeline/checkpoint_to_model.py",
               "pipeline/eval_zero_shot.py", "pipeline/modal_finetune.py"]
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty_shipped": bool(git("status", "--porcelain", "--", *shipped))}


@app.local_entrypoint()
def smoke(method: str = "full"):
    """~25 min on one A100, capped at $1.50 (clean stop at ~$0.88): shows s/step and peak memory."""
    print("exit code", smoke_remote.remote(method, _git_state()))


@app.local_entrypoint()
def train(method: str, run: str, budget: float, extra: str = ""):
    """Training run capped at --budget dollars in total (earlier calls count; raise it to continue a run that
    hit its cap); extra = more 11_finetune.py arguments, space-separated."""
    print("exit code", train_remote.remote(method, run, extra.split(), budget, _git_state()))


@app.local_entrypoint()
def score(run: str, budget: float, model: str = TUGSTUGI, adapter: str = "", sets: str = "all", out: str = "finetuned"):
    """Score a model on the test sets with eval_zero_shot.py (greedy, batch 8). --model: HF id or a volume path
    such as checkpoints/m1_full/final; --adapter: a LoRA final/ on the volume; --sets: comma list or 'all'."""
    chosen = TEST_SETS if sets == "all" else sets.split(",")
    unknown = set(chosen) - set(TEST_SETS)
    if unknown:
        sys.exit(f"unknown sets {unknown}; choose from {TEST_SETS}")
    print("exit code", score_remote.remote(run, model, adapter, chosen, out, budget, _git_state()))


def _download(entries, base: str, dest: Path, skip=lambda rel: False) -> int:
    got = 0
    for e in entries:
        rel = e.path.strip("/")[len(base) + 1:]
        if skip(rel):
            continue
        f = dest / rel
        if f.exists() and f.stat().st_size == e.size:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        part = f.with_name(f.name + ".part")
        with part.open("wb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        part.replace(f)
        got += e.size
        print(f"  {rel} ({e.size / 1e6:.1f} MB)")
    return got


@app.local_entrypoint()
def pull(run: str, with_checkpoint: bool = False, logs_only: bool = False):
    """Download a training run to checkpoints/modal/<run>/: logs + final/ (or partial/); checkpoint/ only if asked
    or if no scorable model exists. Files already downloaded with the right size are skipped (resumable)."""
    from modal.volume import FileEntryType

    base = f"checkpoints/{run}"
    entries = [e for e in vol.listdir(base, recursive=True) if e.type == FileEntryType.FILE]
    names = {e.path.strip("/")[len(base) + 1:] for e in entries}
    has_model = any(n.startswith(("final/", "partial/")) for n in names)
    keep_ck = with_checkpoint or not has_model

    def skip(rel):
        if rel.startswith(("checkpoint.tmp/", "checkpoint.old/")) or (rel.startswith("checkpoint/") and not keep_ck):
            return True
        return logs_only and "/" in rel and not rel.startswith("dev_predictions/")

    dest = Path("checkpoints/modal") / run
    got = _download(entries, base, dest, skip)
    print(f"downloaded {got / 1e9:.2f} GB -> {dest}" + ("" if has_model else
          "  (no final/ or partial/: make one with  pipeline/checkpoint_to_model.py --out_dir " + str(dest) + ")"))


@app.local_entrypoint()
def pull_results(run: str, out: str = "finetuned"):
    """Download a run's scores (every set) to results/modal/<out>/<set>/<run>/ (small files)."""
    from modal.volume import FileEntryType

    base = f"results/{out}"
    entries = [e for e in vol.listdir(base, recursive=True) if e.type == FileEntryType.FILE
               and (f"/{run}/" in "/" + e.path.strip("/") + "/" or e.path.strip("/").endswith(f"_spend/{run}.json"))]
    got = _download(entries, base, Path("results/modal") / out)
    print(f"downloaded {got / 1e6:.1f} MB -> results/modal/{out}/")
