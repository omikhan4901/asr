#!/usr/bin/env python3
"""
Export the four test sets as the exact audio we score, for transcription on Kaggle (CPU only).

Every clip goes through load_row_audio (the same loader pipeline/eval_zero_shot.py uses), so
long-recording slices are cut, mp3 is decoded and everything is resampled to 16 kHz mono exactly
as for our own models, then stored as 16-bit FLAC. Per set it writes
  <out_dir>/<set>/manifest.csv   utt_id, group, ref, audio, duration   (group/ref as eval_zero_shot reads them)
  <out_dir>/<set>/audio/<utt_id>.flac
  <out_dir>/zips/<set>.zip       the same folder, stored (FLAC is already compressed)
Re-running skips clips that already exist, so it is resumable.

  ./Scripts/python.exe pipeline/export_kaggle.py [--sets splits_fa-eval,fleurs-test,...] [--no_zip]
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the GPU belongs to the training runs; this script never needs it

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bangla_asr.common.audio import load_row_audio

SETS = {  # set name = results/zeroshot/<set name>, as eval_zero_shot.py derives it
    "splits_fa-eval": "data/splits_fa/eval.csv",
    "fleurs-test": "data/external/fleurs/test.csv",
    "kathbath-valid": "data/external/kathbath/valid.csv",
    "cv-test": "data/external/cv/test.csv",
    # DL Sprint training clips (cleaned text): only for learning another model's rewrite rules.
    "splits_fa-train": "data/splits_fa/train_clean.csv",
}


def export_set(name, manifest, out_root, limit):
    df = pd.read_csv(manifest, encoding="utf-8-sig")
    if limit:
        df = df.head(limit)
    assert df["utt_id"].is_unique, f"{manifest}: duplicate utt_id"
    ref_col = "text" if "text" in df.columns else "gt"
    group_col = "recording" if "recording" in df.columns else ("group" if "group" in df.columns else "utt_id")
    set_dir = Path(out_root, name)
    (set_dir / "audio").mkdir(parents=True, exist_ok=True)

    rows, hours = [], 0.0
    for r in tqdm(df.to_dict("records"), desc=name):
        dst = set_dir / "audio" / f"{r['utt_id']}.flac"
        if not dst.exists() or dst.stat().st_size == 0:
            audio = load_row_audio(r)
            tmp = dst.with_suffix(".tmp.flac")
            sf.write(str(tmp), audio, 16000, format="FLAC", subtype="PCM_16")
            tmp.replace(dst)
        info = sf.info(str(dst))
        hours += info.duration / 3600
        rows.append({"utt_id": r["utt_id"], "group": str(r[group_col]), "ref": str(r[ref_col]),
                     "audio": f"audio/{r['utt_id']}.flac", "duration": round(info.duration, 3)})
    pd.DataFrame(rows).to_csv(set_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"{name}: {len(rows)} clips, {hours:.2f} h")
    return set_dir


def zip_set(set_dir, zip_dir):
    zip_dir.mkdir(parents=True, exist_ok=True)
    zpath = zip_dir / f"{set_dir.name}.zip"
    tmp = zpath.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.write(set_dir / "manifest.csv", f"{set_dir.name}/manifest.csv")
        for f in sorted((set_dir / "audio").glob("*.flac")):
            z.write(f, f"{set_dir.name}/audio/{f.name}")
    tmp.replace(zpath)
    print(f"  -> {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", default="splits_fa-eval,fleurs-test,kathbath-valid,cv-test")
    p.add_argument("--out_dir", default="data/kaggle_hishab")
    p.add_argument("--no_zip", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N rows per set (smoke test)")
    args = p.parse_args()
    for name in args.sets.split(","):
        set_dir = export_set(name, SETS[name], args.out_dir, args.limit)
        if not args.no_zip:
            zip_set(set_dir, Path(args.out_dir, "zips"))


if __name__ == "__main__":
    main()
