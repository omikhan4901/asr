#!/usr/bin/env python3
"""
SUBAK.KO test split (Hugging Face SUST-CSE-Speech/SUBAK.KO, CC-BY-4.0) as
16 kHz mono FLAC, a manifest, and a zip in the layout Modal unpacks.

Reads data/test-0000{0..4}-of-00005.parquet in file order. group is the
recording: the file name with a trailing -<digits> removed. An existing
FLAC is left in place, so a stopped run can continue.

Example:
  PYTHONPATH=src .venv/Scripts/python.exe pipeline/data/prepare_subakko.py --limit 30 --out_root /tmp/subakko_probe
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "SUST-CSE-Speech/SUBAK.KO"
PARQUETS = [f"data/test-{i:05d}-of-00005.parquet" for i in range(5)]
SR = 16000
COLUMNS = ["utt_id", "audio_path", "text", "group", "source_id", "duration"]


def recording_group(file_path: str) -> str:
    return re.sub(r"-\d+$", "", Path(file_path).stem)


def to_16k_mono(blob: bytes) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(data, dtype=np.float32)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default="data")
    p.add_argument("--limit", type=int, default=0, help="stop after this many clips (0 = the whole test split)")
    args = p.parse_args()

    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_root = Path(args.out_root)
    audio_dir = out_root / "external" / "subakko" / "test"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped, total_s, k = [], 0, 0.0, 0
    stop = False
    for fname in PARQUETS:
        print(f"reading {fname}")
        local = hf_hub_download(REPO, fname, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16, columns=["audio", "transcription", "file_path"]):
            for rec in batch.to_pylist():
                utt_id = f"subakko_test_{k:05d}"
                k += 1
                text = rec["transcription"]
                if text is None or str(text).strip() == "":
                    skipped += 1
                    continue
                dest = audio_dir / f"{utt_id}.flac"
                if dest.is_file():
                    n_samples = sf.info(str(dest)).frames
                else:
                    blob = (rec.get("audio") or {}).get("bytes")
                    if not blob:
                        raise SystemExit(f"{utt_id}: audio bytes missing")
                    audio = to_16k_mono(blob)
                    n_samples = len(audio)
                    tmp = dest.with_suffix(".tmp.flac")
                    sf.write(str(tmp), audio, SR, format="FLAC", subtype="PCM_16")
                    os.replace(tmp, dest)
                dur_s = n_samples / SR
                total_s += dur_s
                file_path = rec["file_path"]
                rows.append({
                    "utt_id": utt_id,
                    "audio_path": f"data/external/subakko/test/{utt_id}.flac",
                    "text": text,
                    "group": recording_group(file_path),
                    "source_id": file_path,
                    "duration": round(dur_s, 3),
                })
                if args.limit and len(rows) >= args.limit:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    manifest = out_root / "external" / "subakko" / "test.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(manifest, index=False, encoding="utf-8-sig")
    zip_path = out_root / "modal" / "subakko-test.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as z:
        for r in rows:
            z.write(audio_dir / f"{r['utt_id']}.flac", f"subakko-test/audio/{r['utt_id']}.flac")
    os.replace(tmp_zip, zip_path)

    report = {
        "clips": len(rows),
        "hours": round(total_s / 3600, 6),
        "recordings": len({r["group"] for r in rows}),
        "clips_skipped_empty_text": skipped,
    }
    report_path = out_root / "external" / "subakko" / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"clips {report['clips']}, hours {report['hours']:.4f}, recordings {report['recordings']}, "
        f"clips skipped for empty text {report['clips_skipped_empty_text']}"
    )
    print(f"manifest {manifest}")
    print(f"zip {zip_path}")
    if rows:
        print("manifest row:", json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
