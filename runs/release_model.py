#!/usr/bin/env python3
"""
Build, verify and publish the final model (0.25 M1 + 0.75 M5-300 weight average, README §5.12) on Hugging Face.

  build    download M1 and M5-300 from the private hub and average them with pipeline/make_soup.py (CPU, ~2 min)
  verify   transcribe the first --n FLEURS test clips on the CPU with the scoring harness and compare with the
           Kaggle T4 predictions of the same model (results/kaggle/finetuned/fleurs-test/<soup>/predictions.jsonl)
  upload   create the public model repo (Apache-2.0) and upload the model folder with docs/model_card.md as README.md

  PYTHONPATH=src .venv/bin/python runs/release_model.py build
  PYTHONPATH=src .venv/bin/python runs/release_model.py verify --n 20
  PYTHONPATH=src .venv/bin/python runs/release_model.py upload --repo OmiKhan/bangla-asr-whisper-medium
Needs `hf auth login` with a token that can read OmiKhan/bangla-asr-hub and write models.
"""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HUB = "OmiKhan/bangla-asr-hub"
MEMBERS = "m1_full+m5_300+m5_300+m5_300"
SOUP = f"soup_{MEMBERS}_t4"
WORK = ROOT / "data" / "release"
MODEL = WORK / "model"


def build():
    from huggingface_hub import snapshot_download
    hub = Path(snapshot_download(HUB, repo_type="dataset", allow_patterns=["models/m1_full/*", "models/m5_300/*"],
                                 local_dir=WORK / "hub"))
    subprocess.run([sys.executable, str(ROOT / "pipeline/make_soup.py"), "--members", MEMBERS,
                    "--models_dir", str(hub / "models"), "--out", str(MODEL)], check=True)
    print("files:", sorted(p.name for p in MODEL.iterdir()))


def verify(n: int):
    from huggingface_hub import hf_hub_download
    z = Path(hf_hub_download(HUB, "zips/fleurs-test.zip", repo_type="dataset", local_dir=WORK / "hub"))
    audio = WORK / "verify" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "data/external/fleurs/test.csv", encoding="utf-8-sig").head(n)
    with zipfile.ZipFile(z) as zf:
        names = {Path(m).stem: m for m in zf.namelist() if not m.endswith("/")}
        for u in man.utt_id:
            src = names[u]
            dst = audio / Path(src).name
            if not dst.exists():
                with zf.open(src) as fi, open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            man.loc[man.utt_id == u, "audio_path"] = str(dst)
    vman = WORK / "verify" / "fleurs-test.csv"
    man.to_csv(vman, index=False)
    out = WORK / "verify" / "results"
    subprocess.run([sys.executable, str(ROOT / "pipeline/eval_zero_shot.py"), "--manifest", str(vman), "--model", str(MODEL),
                    "--name", "release_cpu", "--set_name", "fleurs-test", "--out_dir", str(out)], check=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    local = {r["utt_id"]: r["hyp"] for r in map(json.loads, open(out / "fleurs-test/release_cpu/predictions.jsonl", encoding="utf-8"))}
    kaggle = {r["utt_id"]: r["hyp"] for r in map(json.loads, open(ROOT / f"results/kaggle/finetuned/fleurs-test/{SOUP}/predictions.jsonl", encoding="utf-8"))}
    same = sum(local[u] == kaggle[u] for u in local)
    print(f"identical transcripts: {same} / {len(local)} (CPU fp32 here vs T4 fp16 on Kaggle)")
    for u in local:
        if local[u] != kaggle[u]:
            print(f"  {u}\n    cpu   : {local[u]}\n    kaggle: {kaggle[u]}")


def upload(repo: str):
    from huggingface_hub import HfApi
    api = HfApi()
    shutil.copy(ROOT / "docs/model_card.md", MODEL / "README.md")
    shutil.copy(ROOT / "LICENSE", MODEL / "LICENSE")
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(MODEL), commit_message="Final model: 0.25 M1 + 0.75 M5-300")
    print("published:", f"https://huggingface.co/{repo}", sorted(f for f in api.list_repo_files(repo)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "verify", "upload"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--repo", default="OmiKhan/bangla-asr-whisper-medium")
    a = ap.parse_args()
    {"build": build, "verify": lambda: verify(a.n), "upload": lambda: upload(a.repo)}[a.step]()
