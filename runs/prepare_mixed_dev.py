#!/usr/bin/env python3
"""
Mixed development set for choosing between model variants (weight averages of M5-300 with other models):
DL Sprint dev (exists) + Common Voice dev (2,000 random clips, seed 0) + FLEURS validation. Used only to pick a
variant, never as a test set. Kathbath has no clean dev split (every Kathbath train speaker is in training).

Steps (all resumable):
  1. CV dev manifest (pipeline/data/prepare_cv.py --split dev), 2,000 clips sampled with seed 0 -> data/external/cv/dev.csv
  2. FLEURS validation (pipeline/data/prepare_hf.py --preset fleurs --split validation) -> data/external/fleurs/validation.csv
  3. drop clips whose sentence (normalized) occurs in any training text of the compared models or in a test set
  4. FLAC export + zip (pipeline/export_kaggle.py layout) -> data/kaggle_hishab/zips/{cv-dev,fleurs-validation}.zip
  5. with --upload: zips + Kaggle manifests to the hub (zips/<set>.zip, manifests/external/<corpus>/<split>.csv)

  PYTHONPATH=src ./Scripts/python.exe runs/prepare_mixed_dev.py            # build, report, no upload
  PYTHONPATH=src ./Scripts/python.exe runs/prepare_mixed_dev.py --upload   # build (skips finished parts) + upload
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

HUB = "OmiKhan/bangla-asr-hub"
OUT = ROOT / "data" / "kaggle_hishab"
SETS = {"cv-dev": ("data/external/cv/dev.csv", "external/cv/dev.csv"),
        "fleurs-validation": ("data/external/fleurs/validation.csv", "external/fleurs/validation.csv")}
TESTS = ["data/splits_fa/eval.csv", "data/splits_fa/dev.csv", "data/external/fleurs/test.csv",
         "data/external/kathbath/valid.csv", "data/external/cv/test.csv"]
N_CV = 2000


def run(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def texts(path, col=None) -> set[str]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    col = col or ("text" if "text" in df.columns else "gt")
    return {normalize_bangla_text(str(t)) for t in df[col]}


def training_texts() -> dict[str, set[str]]:
    """Sentences every compared model may have seen in training (M1: DL Sprint; M5-full / M5-300: + CV, Kathbath,
    Shrutilipi train, OpenSLR 53 = every sentence of the corpus, a superset of the joined training samples)."""
    from huggingface_hub import hf_hub_download

    out = {"DL Sprint train": texts(ROOT / "data/splits_fa/train_clean.csv")}
    for n in ("cv-train", "kathbath-train", "shrutilipi-train"):
        out[n] = texts(hf_hub_download(HUB, f"manifests/matrix/{n}.csv", repo_type="dataset"))
    slr = pd.read_csv(ROOT / "data/openslr53/utt_spk_text.tsv", sep="\t", header=None, quoting=csv.QUOTE_NONE, dtype=str)
    out["OpenSLR 53"] = {normalize_bangla_text(str(t)) for t in slr[2]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    cv = ROOT / SETS["cv-dev"][0]
    if not cv.exists():
        run("pipeline/data/prepare_cv.py", "--split", "dev", "--out_dir", "data/external/cv/_dev_all")
        full = pd.read_csv(ROOT / "data/external/cv/_dev_all/dev.csv", encoding="utf-8-sig")
        full.sample(n=N_CV, random_state=0).to_csv(cv, index=False, encoding="utf-8-sig")
    fl = ROOT / SETS["fleurs-validation"][0]
    if not fl.exists():
        run("pipeline/data/prepare_hf.py", "--preset", "fleurs", "--split", "validation")

    # 3. sentence overlap with training text and test sets -> those clips are dropped (once; recorded in the file)
    seen = training_texts()
    seen["test sets"] = set().union(*(texts(ROOT / t) for t in TESTS))
    for name, (local, _) in SETS.items():
        df = pd.read_csv(ROOT / local, encoding="utf-8-sig")
        norm = df["text"].map(lambda t: normalize_bangla_text(str(t)))
        hits = {k: int(norm.isin(v).sum()) for k, v in seen.items()}
        drop = norm.isin(set().union(*seen.values()))
        print(f"{name}: {len(df)} clips; sentence also in {hits}; dropping {int(drop.sum())}")
        if drop.any():
            df[~drop].to_csv(ROOT / local, index=False, encoding="utf-8-sig")

    # 4. FLAC export + zip, the same layout Kaggle unpacks for every other set
    import export_kaggle
    for name, (local, _) in SETS.items():
        set_dir = export_kaggle.export_set(name, str(ROOT / local), str(OUT), None)
        export_kaggle.zip_set(set_dir, OUT / "zips")

    # Kaggle manifests: audio_path -> the unpacked FLAC (data/clips/<set>/audio/<utt>.flac), as modal_finetune.build_manifest
    hub_man = {}
    for name, (local, hub_path) in SETS.items():
        df = pd.read_csv(ROOT / local, encoding="utf-8-sig")
        df["audio_path"] = [f"data/clips/{name}/audio/{u}.flac" for u in df["utt_id"]]
        for col in ("start", "end"):
            if col in df.columns:
                df[col] = None
        p = OUT / "hub_manifests" / hub_path
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        hub_man[name] = (p, hub_path)
        print(f"{name}: {len(df)} clips, {df['duration'].sum() / 3600:.2f} h, {df['group'].nunique()} speakers/groups")

    if not a.upload:
        print("\nbuilt; run again with --upload to send it to the hub")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    for name, (p, hub_path) in hub_man.items():
        api.upload_file(path_or_fileobj=str(OUT / "zips" / f"{name}.zip"), path_in_repo=f"zips/{name}.zip",
                        repo_id=HUB, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"manifests/{hub_path}", repo_id=HUB, repo_type="dataset")
        print(f"uploaded {name}")
    print("done: mixed dev set on the hub")


if __name__ == "__main__":
    main()
