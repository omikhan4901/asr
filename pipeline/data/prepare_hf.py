#!/usr/bin/env python3
"""
Download a Bangla split of a Hugging Face speech dataset (parquet) and convert
it to 16 kHz mono WAV files + a manifest CSV that every other script reads.

Presets:
  fleurs    google/fleurs, config bn_in        splits: train, validation, test
  kathbath  ai4bharat/Kathbath, bengali/       splits: train, valid   (accept terms on HF first)
  shrutilipi ai4bharat/Shrutilipi, bengali/    splits: train          (accept terms on HF first)

Output: data/external/<preset>/<split>/*.wav and data/external/<preset>/<split>.csv
with columns utt_id, audio_path, text, group, duration.
Resumable: existing WAV files are not rewritten.

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/data/prepare_hf.py --preset fleurs --split test
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PRESETS = {
    "fleurs": {"repo": "google/fleurs", "prefix": "parquet-data/bn_in/",
               "text": ["raw_transcription", "transcription"], "group": ["id"]},
    "kathbath": {"repo": "ai4bharat/Kathbath", "prefix": "bengali/",
                 "text": ["text", "transcript", "sentence", "transcription"], "group": ["speaker_id", "speaker", "client_id"]},
    "shrutilipi": {"repo": "ai4bharat/Shrutilipi", "prefix": "bengali/",
                   "text": ["transcriptions", "text", "transcript", "sentence"], "group": ["speaker_id", "speaker", "source"]},
}
AUDIO_COLS = ["audio", "audio_filepath", "audio_file"]


def decode(cell, target_sr=16000):
    import soundfile as sf

    if isinstance(cell, dict):
        if cell.get("bytes"):
            data, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif cell.get("array") is not None:
            data, sr = np.asarray(cell["array"], dtype=np.float32), int(cell.get("sampling_rate", target_sr))
        else:
            raise ValueError("audio cell has neither bytes nor array")
    else:
        data, sr = sf.read(io.BytesIO(cell), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"Could not find a {what} column among {cols}. Please report this line.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out_root", default="data/external")
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows (testing)")
    p.add_argument("--max_hours", type=float, default=None, help="Stop after this many hours of audio (training subsets)")
    args = p.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    cfg = PRESETS[args.preset]
    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                   if f.startswith(cfg["prefix"]) and Path(f).name.startswith(args.split + "-") and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"No parquet files for split '{args.split}' under {cfg['repo']}/{cfg['prefix']}")
    print(f"{cfg['repo']}: {len(files)} parquet file(s) for split '{args.split}'")

    out_dir = Path(args.out_root, args.preset, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, total_s, n = [], 0.0, 0
    for fname in files:
        local = hf_hub_download(cfg["repo"], fname, repo_type="dataset")
        pf = pq.ParquetFile(local)
        cols = pf.schema_arrow.names
        acol, tcol = pick(cols, AUDIO_COLS, "audio"), pick(cols, cfg["text"], "text")
        gcol = next((c for c in cfg["group"] if c in cols), None)
        id_col = next((c for c in ["id", "utt_id", "audio_id", "path"] if c in cols), None)
        if n == 0:
            print(f"columns: {cols}\nusing audio={acol} text={tcol} group={gcol} id={id_col}")
        for batch in pf.iter_batches(batch_size=64):
            for rec in batch.to_pylist():
                uid = f"{args.preset}_{args.split}_{n:06d}"
                wav = out_dir / f"{uid}.wav"
                if wav.exists():
                    dur = sf.info(str(wav)).duration
                else:
                    audio = decode(rec[acol])
                    tmp = wav.with_name(wav.name + ".tmp")
                    sf.write(str(tmp), audio, 16000, format="WAV")
                    os.replace(tmp, wav)
                    dur = len(audio) / 16000
                text = rec[tcol]
                if isinstance(text, list):
                    text = " ".join(map(str, text))
                rows.append({"utt_id": uid, "audio_path": str(wav).replace("\\", "/"), "text": str(text),
                             "group": str(rec[gcol]) if gcol else uid,
                             "source_id": str(rec[id_col]) if id_col else "", "duration": round(dur, 3)})
                n += 1
                total_s += dur
                if (args.limit and n >= args.limit) or (args.max_hours and total_s >= args.max_hours * 3600):
                    break
            else:
                continue
            break
        else:
            continue
        break

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_root, args.preset, f"{args.split}.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(df)} utterances, {total_s / 3600:.2f} h -> {out_dir}  (manifest: {args.preset}/{args.split}.csv)")


if __name__ == "__main__":
    main()
