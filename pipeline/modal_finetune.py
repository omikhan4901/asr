#!/usr/bin/env python3
"""
Run pipeline/11_finetune.py on a Modal GPU (Phase 4: full fine-tune, encoder-only, matrix runs).

The training code is the same file as on the RTX 3060, called with the same arguments
(same manifest paths, so config.json records the same recipe). Only the audio source
differs: clips exported by pipeline/export_kaggle.py (16 kHz FLAC, bit-identical to the
slices load_row_audio cuts from the raw recordings) instead of the raw WAVs, so the
Modal manifests carry the FLAC path and no start/end. Text and every other column are
copied unchanged from data/splits_fa/{train_clean,dev}.csv.

Volume "bangla-asr-ft":
  /zips/<set>.zip              exported clips (uploaded once)
  /manifests/<file>.csv        training and dev manifests pointing at the FLAC clips
  /checkpoints/<run>/          everything 11_finetune.py writes (resumable)
  /hf/                         Hugging Face cache (tugstugi downloaded once)

Steps (Git Bash, repo root; prefix every command with  PYTHONIOENCODING=utf-8):
  1. ./Scripts/python.exe pipeline/export_kaggle.py --sets splits_fa-train,splits_fa-dev
  2. ./Scripts/modal.exe run pipeline/modal_finetune.py::upload
  3. ./Scripts/modal.exe run --detach pipeline/modal_finetune.py::smoke --method full
  4. ./Scripts/modal.exe run --detach pipeline/modal_finetune.py::train --method full --run m1_full --budget 8
  5. ./Scripts/modal.exe run pipeline/modal_finetune.py::pull --run m1_full     (--logs-only: no weights)
--detach keeps the job running when this terminal closes or the internet drops.

Money safety: --budget caps the whole run in dollars (all calls and restarts, tracked in
<run>/modal_spend.json at the price below). Before the cap the trainer gets one Ctrl+C,
finishes its step and saves; the checkpoint is then exported to <run>/partial/ (a model
that can be scored). At the cap it is killed; the last periodic checkpoint (every 100
steps) stays on the volume, and pull + pipeline/checkpoint_to_model.py make it usable
locally. To continue a stopped run, run step 4 again with a higher --budget.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import modal

APP = "bangla-asr-finetune"
VOLUME = "bangla-asr-ft"
GPU = "A100-40GB"  # full AdamW state of Whisper-medium (~12 GB) + activations; 80 GB not needed
ZIPS = {"splits_fa-train": "data/kaggle_hishab/zips/splits_fa-train.zip",
        "splits_fa-dev": "data/kaggle_hishab/zips/splits_fa-dev.zip"}
# manifest as 11_finetune.py is called -> (source manifest, exported set)
MANIFESTS = {"train_clean.csv": ("data/splits_fa/train_clean.csv", "splits_fa-train"),
             "dev.csv": ("data/splits_fa/dev.csv", "splits_fa-dev")}
REMOTE_PROJ, WORK = "/root/proj", "/root/work"
# Cost per hour of one container (modal.com/pricing, 2026-10-10): A100 40 GB $2.10 + 4 physical cores x $0.047
# + 24 GiB x $0.008. Used only for the budget cap; the Modal dashboard shows the billed amount.
RATE_PER_H = 2.10 + 4 * 0.047 + 24 * 0.008
RESERVE_MIN = 15  # minutes kept back for the last dev check, the checkpoint save and the partial/ export

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
# Same library versions as the local venv (provenance: run_info.json records them).
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.13.0", "transformers==5.15.0", "peft==0.21.2", "accelerate==1.14.0",
                 "safetensors==0.8.0", "numpy==2.5.2", "pandas==3.0.5", "soundfile==0.14.0",
                 "rapidfuzz==3.14.5")
    .env({"PYTHONPATH": f"{REMOTE_PROJ}/src", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
          "HF_HOME": "/vol/hf"})
    .add_local_dir("src/bangla_asr", f"{REMOTE_PROJ}/src/bangla_asr", ignore=["**/__pycache__"])
    .add_local_file("pipeline/11_finetune.py", f"{REMOTE_PROJ}/pipeline/11_finetune.py")
    .add_local_file("pipeline/checkpoint_to_model.py", f"{REMOTE_PROJ}/pipeline/checkpoint_to_model.py")
)


def build_manifests(out_dir: Path) -> list[Path]:
    """Modal copies of the manifests: audio_path -> exported FLAC, start/end emptied; checked against the zips."""
    import pandas as pd

    paths = []
    for name, (src, set_name) in MANIFESTS.items():
        df = pd.read_csv(src, encoding="utf-8-sig")
        with zipfile.ZipFile(ZIPS[set_name]) as z:
            in_zip = {Path(n).stem for n in z.namelist() if n.endswith(".flac")}
        missing = set(df["utt_id"]) - in_zip
        if missing or len(in_zip) != len(df):
            sys.exit(f"{ZIPS[set_name]} does not match {src}: {len(missing)} clips missing, "
                     f"{len(in_zip)} in zip vs {len(df)} rows. Re-run pipeline/export_kaggle.py --sets {set_name}")
        df["audio_path"] = [f"data/clips/{set_name}/audio/{u}.flac" for u in df["utt_id"]]
        df["start"], df["end"] = None, None  # whole clip; load_row_audio falls back to load_audio
        out = out_dir / name
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} rows -> FLAC clips of {set_name}")
        paths.append(out)
    return paths


@app.local_entrypoint()
def upload():
    """Upload the clip zips (once; skipped when the same size is already there) and the manifests (always)."""
    import tempfile

    have = {e.path.strip("/"): e.size for e in vol.listdir("/zips")} if _exists("/zips") else {}
    with tempfile.TemporaryDirectory() as tmp:
        manifests = build_manifests(Path(tmp))
        with vol.batch_upload(force=True) as batch:
            for m in manifests:
                batch.put_file(str(m), f"/manifests/{m.name}")
            for set_name, local in ZIPS.items():
                size = Path(local).stat().st_size
                if have.get(f"zips/{set_name}.zip") == size:
                    print(f"{set_name}.zip already on the volume, skipped")
                    continue
                print(f"uploading {local} ({size / 1e9:.2f} GB); this can take a while")
                batch.put_file(local, f"/zips/{set_name}.zip")
    print("done:", [(e.path, round(e.size / 1e9, 2)) for e in vol.listdir("/zips")])


def _exists(path):
    try:
        vol.listdir(path)
        return True
    except Exception:
        return False


def _prepare():
    """Unpack the clips to local disk (fast reads) and lay out the working dir as 11_finetune.py expects."""
    import os

    work = Path(WORK)
    (work / "data").mkdir(parents=True, exist_ok=True)
    clips = Path("/tmp/clips")
    for set_name in ZIPS:
        if not (clips / set_name / "manifest.csv").exists():
            with zipfile.ZipFile(f"/vol/zips/{set_name}.zip") as z:
                z.extractall(clips)
            print(f"unpacked {set_name}: {len(list((clips / set_name / 'audio').glob('*.flac')))} clips")
    for link, target in [(work / "data" / "clips", clips), (work / "data" / "splits_fa", Path("/vol/manifests")),
                         (work / "checkpoints", Path("/vol/checkpoints"))]:
        Path(target).mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(target, link)


def _run_finetune(method: str, out_dir: str, extra: list[str], budget: float) -> int:
    """Run 11_finetune.py under a dollar cap for the whole run (all calls, restarts included):
    Ctrl+C (clean save) before the cap, kill at the cap. Spend so far: <out_dir>/modal_spend.json."""
    import json
    import signal
    import threading
    import time

    spend_file = Path("/vol") / out_dir / "modal_spend.json"
    before = json.loads(spend_file.read_text())["dollars"] if spend_file.exists() else 0.0
    t0 = time.time()
    spent = lambda: before + (time.time() - t0) / 3600 * RATE_PER_H  # noqa: E731
    soft = budget - RESERVE_MIN / 60 * RATE_PER_H  # leaves time to finish a dev check, save and convert
    if spent() >= soft:
        sys.exit(f"{out_dir}: ${before:.2f} already spent of a ${budget:.2f} budget "
                 f"(clean stop at ${soft:.2f}). Pass a higher --budget to continue.")
    _prepare()
    cmd = [sys.executable, f"{REMOTE_PROJ}/pipeline/11_finetune.py", "--method", method,
           "--train_manifest", "data/splits_fa/train_clean.csv", "--dev_manifest", "data/splits_fa/dev.csv",
           "--out_dir", out_dir, "--num_workers", "4", *extra]
    print(f"budget ${budget:.2f}: clean stop at ${soft:.2f}, hard stop at ${budget:.2f} "
          f"(${RATE_PER_H:.2f}/h for {GPU} + 4 cores + 24 GiB)", flush=True)
    print("$", " ".join(cmd), flush=True)
    # Own session: a signal sent to the whole container must not reach the trainer twice
    # (a second Ctrl+C makes 11_finetune.py quit without saving); stop() forwards exactly one.
    child = subprocess.Popen(cmd, cwd=WORK, start_new_session=True)
    flag = {"soft": False, "hard": False}

    def stop(*_):
        if not flag["soft"]:
            flag["soft"] = True
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)  # Modal preemption interrupts the container: save, then exit
    signal.signal(signal.SIGTERM, stop)

    def watchdog():
        while child.poll() is None:
            spend_file.parent.mkdir(parents=True, exist_ok=True)
            spend_file.write_text(json.dumps({"dollars": round(spent(), 4), "rate_per_h": RATE_PER_H}))
            if spent() >= soft and not flag["soft"]:
                print(f"[budget] ${spent():.2f} spent: stopping cleanly (finish step, save)", flush=True)
                stop()
            if spent() >= budget and not flag["hard"]:
                flag["hard"] = True
                print(f"[budget] ${spent():.2f} spent: hard stop; the last periodic checkpoint is kept", flush=True)
                child.kill()
            time.sleep(10)

    threading.Thread(target=watchdog, daemon=True).start()
    code = child.wait()
    spend_file.write_text(json.dumps({"dollars": round(spent(), 4), "rate_per_h": RATE_PER_H}))
    vol.commit()
    if code != 0 and not flag["hard"] and (Path(WORK) / out_dir / "checkpoint" / "state.pt").exists():
        # Stopped early: also leave a scorable model, so the work done is usable even if nothing runs again.
        subprocess.run([sys.executable, f"{REMOTE_PROJ}/pipeline/checkpoint_to_model.py", "--out_dir", out_dir],
                       cwd=WORK)
        vol.commit()
    print(f"exit code {code} (0 = finished: final/; 3 = stopped early and saved: partial/; "
          f"run the same command again to continue) | spent about ${spent():.2f}", flush=True)
    return code


@app.function(image=image, gpu=GPU, cpu=4, memory=24576, volumes={"/vol": vol}, timeout=12 * 3600)
def train_remote(method: str, run: str, extra: list[str], budget: float) -> int:
    return _run_finetune(method, f"checkpoints/{run}", extra, budget)


@app.function(image=image, gpu=GPU, cpu=4, memory=24576, volumes={"/vol": vol}, timeout=40 * 60)
def smoke_remote(method: str) -> int:
    # Fresh smoke dir each time: the measured speed and memory must come from this image.
    subprocess.run(["rm", "-rf", f"/vol/checkpoints/smoke_{method}"])
    # No --max_minutes: the budget watchdog stops it (clean stop at ~$0.88, about 21 min after start), so
    # the smoke test also exercises the stop path, the checkpoint save and the partial/ export.
    return _run_finetune(method, f"checkpoints/smoke_{method}",
                         ["--dev_limit", "64", "--eval_every", "40", "--memory_probe"], budget=1.5)


@app.local_entrypoint()
def smoke(method: str = "full"):
    """~25 min on one A100, capped at $1.50 (clean stop at ~$0.88): image, model download, unpack, training
    until the budget stop, save, partial/ export. Shows s/step and peak memory for the cost estimate."""
    print("exit code", smoke_remote.remote(method))


@app.local_entrypoint()
def train(method: str, run: str, budget: float, extra: str = ""):
    """Training run capped at --budget dollars in total for this run (earlier calls count; raise it to
    continue a run that hit its cap); extra = more 11_finetune.py arguments, space-separated."""
    print("exit code", train_remote.remote(method, run, extra.split(), budget))


@app.local_entrypoint()
def pull(run: str, with_checkpoint: bool = False, logs_only: bool = False):
    """Download a run to checkpoints/modal/<run>/: logs + final/ (or partial/); checkpoint/ only if asked or
    if no scorable model exists. Files already downloaded with the right size are skipped (resumable)."""
    from modal.volume import FileEntryType

    base = f"checkpoints/{run}"
    entries = [e for e in vol.listdir(base, recursive=True) if e.type == FileEntryType.FILE]
    names = {e.path.strip("/")[len(base) + 1:] for e in entries}
    has_model = any(n.startswith(("final/", "partial/")) for n in names)
    keep_ck = with_checkpoint or not has_model
    dest = Path("checkpoints/modal") / run
    got = 0
    for e in entries:
        rel = e.path.strip("/")[len(base) + 1:]
        if rel.startswith(("checkpoint.tmp/", "checkpoint.old/")) or (rel.startswith("checkpoint/") and not keep_ck):
            continue
        if logs_only and "/" in rel and not rel.startswith("dev_predictions/"):
            continue  # logs, config, spend, run_info and dev predictions only (no weights)
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
    print(f"downloaded {got / 1e9:.2f} GB -> {dest}" + ("" if has_model else
          "  (no final/ or partial/: make one with  pipeline/checkpoint_to_model.py --out_dir " + str(dest) + ")"))
