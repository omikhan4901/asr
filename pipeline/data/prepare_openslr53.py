#!/usr/bin/env python3
"""
OpenSLR 53 ("Large Bengali ASR training data set", Google, CC BY-SA 4.0; Kjartansson et al. 2018) as a
training source for M5-300: 218,703 read utterances, 508 speakers, 16 kHz FLAC, mean 3.3 s (~203 h).

Whisper pads every input to 30 s, so training cost follows the number of clips, not hours: consecutive
utterances of ONE speaker are joined (0.3 s silence between them) into samples of at most --pack_seconds,
which cuts ~155k clips to ~21k for ~144 h. Same text rules as the other sources (clean_train_text.py);
every utterance whose normalized text equals a test-set sentence is dropped (11 Common Voice sentences).
Speakers are taken in a fixed random order (seed 0) until --hours is reached.

  python pipeline/data/prepare_openslr53.py --download --raw /vol/openslr53_raw --out /tmp/matrix --hours 144 \
      --exclude_text /vol/manifests/eval.csv /vol/manifests/dev.csv /vol/manifests/external/fleurs/test.csv ...
Output: <out>/openslr53-train/{audio/*.flac, manifest.csv, report.json} and <out>/zips/openslr53-train.zip
(layout of pipeline/data/prepare_matrix.py). Resumable: shards already downloaded are kept.
"""

import argparse
import csv
import io
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_matrix import clean, write_manifest, zip_folder  # noqa: E402


def _load_normalizer():
    """bangla_asr.common.text loaded as a single file: the package __init__ imports torch, which the Modal
    data image does not have (same rule as af3ba1c)."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "src" / "bangla_asr" / "common" / "text.py"
    spec = importlib.util.spec_from_file_location("bangla_text", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_bangla_text


normalize_bangla_text = _load_normalizer()

BASE = "https://www.openslr.org/resources/53"
SHARDS = "0123456789abcdef"
NAME, SR, SEED = "openslr53-train", 16000, 0


def download(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    for name in ["utt_spk_text.tsv"] + [f"asr_bengali_{s}.zip" for s in SHARDS]:
        dst = raw / name
        size = int(urllib.request.urlopen(urllib.request.Request(f"{BASE}/{name}", method="HEAD")).headers["Content-Length"])
        if dst.exists() and dst.stat().st_size == size:
            continue
        print(f"downloading {name} ({size / 1e9:.2f} GB)", flush=True)
        tmp = dst.with_name(name + ".part")
        urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
        assert tmp.stat().st_size == size, f"{name}: incomplete download"
        tmp.replace(dst)


def test_sentences(paths):
    out = set()
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig")
        col = "text" if "text" in df.columns else "gt"
        out |= {normalize_bangla_text(str(t)) for t in df[col]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder with utt_spk_text.tsv and asr_bengali_<x>.zip")
    ap.add_argument("--download", action="store_true", help="fetch the 16 shards (~14.7 GB) + tsv into --raw first")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=144.0)
    ap.add_argument("--pack_seconds", type=float, default=25.0)
    ap.add_argument("--gap", type=float, default=0.3)
    ap.add_argument("--exclude_text", nargs="*", default=[], help="test manifests: their sentences are dropped")
    ap.add_argument("--delete_raw", action="store_true", help="delete the shards after joining, before zipping (disk)")
    args = ap.parse_args()
    if args.download:
        download(args.raw)

    df = pd.read_csv(args.raw / "utt_spk_text.tsv", sep="\t", header=None, names=["utt", "spk", "text"],
                     quoting=csv.QUOTE_NONE, dtype=str)
    n0 = len(df)
    df["text"] = df["text"].map(clean)
    df = df.dropna(subset=["text"])
    n_clean = len(df)
    banned = test_sentences(args.exclude_text)
    df = df[~df["text"].map(normalize_bangla_text).isin(banned)]
    n_test = n_clean - len(df)
    zips = {s: zipfile.ZipFile(args.raw / f"asr_bengali_{s}.zip") for s in SHARDS if (args.raw / f"asr_bengali_{s}.zip").exists()}
    names = {s: set(z.namelist()) for s, z in zips.items()}
    member = lambda u: f"asr_bengali/data/{u[:2]}/{u}.flac"
    df = df[df["utt"].map(lambda u: u[0] in names and member(u) in names[u[0]])]

    speakers = sorted(df["spk"].unique())
    random.Random(SEED).shuffle(speakers)
    by_spk = {k: g.sort_values("utt") for k, g in df.groupby("spk")}
    audio_dir = args.out / NAME / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(args.gap * SR), dtype=np.float32)
    rows, total, used_utts, used_spk = [], 0.0, 0, 0

    def flush(spk, parts, texts):
        nonlocal total
        i = len(rows)
        wav = np.concatenate([x for p in parts for x in (p, gap)][:-1])
        path = audio_dir / f"openslr53_train_{i:06d}.flac"
        if not path.exists():
            sf.write(path, wav, SR, format="FLAC")
        dur = len(wav) / SR
        rows.append({"utt_id": path.stem, "audio_path": f"data/clips/{NAME}/audio/{path.name}", "text": " ".join(texts),
                     "group": spk, "duration": round(dur, 3)})
        total += dur

    for spk in speakers:
        if total >= args.hours * 3600:
            break
        used_spk += 1
        parts, texts, cur = [], [], 0.0
        for u, t in zip(by_spk[spk]["utt"], by_spk[spk]["text"]):
            wav, sr = sf.read(io.BytesIO(zips[u[0]].read(member(u))), dtype="float32")
            assert sr == SR, f"{u}: {sr} Hz"
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            d = len(wav) / SR
            if parts and cur + args.gap + d > args.pack_seconds:
                flush(spk, parts, texts)
                parts, texts, cur = [], [], 0.0
            parts.append(wav); texts.append(t); cur += d + (args.gap if len(parts) > 1 else 0)
            used_utts += 1
        if parts:
            flush(spk, parts, texts)
    report = {"source": "OpenSLR 53 (CC BY-SA 4.0)", "utterances_in_tsv": n0, "dropped_text_rules": n0 - n_clean,
              "dropped_test_sentences": n_test, "utterances_used": used_utts, "speakers_used": used_spk,
              "pack_seconds": args.pack_seconds, "gap_seconds": args.gap, "seed": SEED}
    for z in zips.values():
        z.close()
    if args.delete_raw:
        for f in args.raw.glob("asr_bengali_*.zip"):
            f.unlink()
    write_manifest(args.out, NAME, rows, report)
    zip_folder(args.out, NAME)


if __name__ == "__main__":
    main()
