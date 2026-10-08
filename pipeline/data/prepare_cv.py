#!/usr/bin/env python3
"""
Manifest for a Common Voice Bengali split, from the extracted corpus.

Reads data/CommonVoiceCorpus/bn/<split>.tsv (official split) and writes
data/external/cv/<split>.csv with columns utt_id, audio_path, text, group,
duration. audio_path points at the original mp3 clip (decoded and resampled
to 16 kHz on the fly by bangla_asr.common.audio.load_audio), so no audio is
copied. group = client_id (speaker), used for the cluster bootstrap.

Caveat for the paper: a public model fine-tuned on an older Common Voice
release may have seen some of these clips (splits change between releases).

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/data/prepare_cv.py --split test
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--cv_dir", default="data/CommonVoiceCorpus/bn")
    p.add_argument("--out_dir", default="data/external/cv")
    args = p.parse_args()

    cv = Path(args.cv_dir)
    df = pd.read_csv(cv / f"{args.split}.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    dur = pd.read_csv(cv / "clip_durations.tsv", sep="\t", dtype={"clip": str})
    dur = dict(zip(dur["clip"], dur["duration[ms]"] / 1000.0))

    missing = [c for c in df["path"] if not (cv / "clips" / c).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing from {cv / 'clips'} (first: {missing[0]})")

    out = pd.DataFrame({
        "utt_id": [f"cv_{args.split}_{Path(c).stem.rsplit('_', 1)[-1]}" for c in df["path"]],
        "audio_path": [(cv / "clips" / c).as_posix() for c in df["path"]],
        "text": df["sentence"].fillna(""),
        "group": df["client_id"],
        "duration": [dur.get(c) for c in df["path"]],
    })
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir, f"{args.split}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(out)} clips, {out['group'].nunique()} speakers, {out['duration'].sum() / 3600:.2f} h -> {path}")


if __name__ == "__main__":
    main()
