#!/usr/bin/env python3
"""
Training data for the Phase 4 matrix runs (M2-M5): one folder + zip + manifest per source,
in the layout pipeline/modal_finetune.py unpacks (zip top folder = set name, manifest
audio_path = data/clips/<set>/...).

  cv          Common Voice bn train (local download, data/CommonVoiceCorpus/bn), original mp3s
              (decoded in the container exactly like cv-test). Speakers of CV dev/test removed.
  kathbath    ai4bharat/Kathbath bengali train (HF, gated), 16 kHz FLAC. Speakers of Kathbath
              valid (our test set) removed. Random --max_hours subset (seed 0) over all shards.
  shrutilipi  ai4bharat/Shrutilipi bengali train (HF, gated), 16 kHz FLAC, random --max_hours subset.
  merged      M5: --hours_each hours drawn at random from each source manifest given with --from
              (a source given as path:H uses H hours instead; --hours_each 0 = all of every source).
              --name sets the list name (m5-merged; Phase 5: m5-full = all 156 h, m5-300 = + OpenSLR 53).

Text: the same deterministic rules as pipeline/data/clean_train_text.py (bracket tags removed,
rows with letters outside the Bengali block dropped). Every row keeps its source text otherwise:
the matrix measures how each corpus's own conventions transfer.

  ./Scripts/python.exe pipeline/data/prepare_matrix.py cv --out data/matrix
  python pipeline/data/prepare_matrix.py kathbath --out /tmp/matrix --max_hours 50 --exclude_groups valid_speakers.txt
  python pipeline/data/prepare_matrix.py merged --out /tmp/matrix --hours_each 10 \
      --from data/splits_fa/train_clean.csv data/splits_fa/matrix/cv-train.csv ...
Outputs: <out>/<set>/manifest.csv (utt_id, audio_path, text, group, duration), <out>/zips/<set>.zip,
<out>/<set>/report.json. Resumable: finished audio files are not rewritten.
"""

import argparse
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_train_text import foreign_letters, strip_tags  # noqa: E402
from prepare_hf import decode  # noqa: E402  (same decoder and resampling as the test-set preparation)

SEED = 0
HF = {"kathbath": ("ai4bharat/Kathbath", "bengali/train-", ["text", "transcript", "sentence"], ["speaker_id", "speaker"]),
      "shrutilipi": ("ai4bharat/Shrutilipi", "bengali/train-", ["transcriptions", "text", "transcript"], ["speaker_id", "speaker", "source"])}


def clean(text) -> str | None:
    t = strip_tags(str(text))
    return None if not t or foreign_letters(t) else t


def pick(cols, names):
    return next((c for c in names if c in cols), None)


def zip_folder(out: Path, name: str):
    """<out>/zips/<name>.zip holding <name>/... (stored: FLAC and mp3 are already compressed)."""
    dst = out / "zips" / f"{name}.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    files = sorted(p for p in (out / name).rglob("*") if p.is_file() and p.name not in ("manifest.csv", "report.json"))
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.relative_to(out).as_posix())
    tmp.replace(dst)
    print(f"{dst}: {len(files)} files, {dst.stat().st_size / 1e9:.2f} GB")


def write_manifest(out: Path, name: str, rows: list[dict], report: dict):
    df = pd.DataFrame(rows, columns=["utt_id", "audio_path", "text", "group", "duration"])
    assert df["utt_id"].is_unique
    df.to_csv(out / name / "manifest.csv", index=False, encoding="utf-8-sig")
    report.update(rows=len(df), hours=round(float(df["duration"].sum()) / 3600, 3), speakers=int(df["group"].nunique()))
    (out / name / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {report}")


def cv(args):
    import shutil

    root, name = Path(args.cv_root), "cv-train"
    dur = pd.read_csv(root / "clip_durations.tsv", sep="\t", quoting=3).set_index("clip")["duration[ms]"] / 1000
    tsv = lambda s: pd.read_csv(root / f"{s}.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)  # noqa: E731
    train, held = tsv("train"), set(tsv("test")["client_id"]) | set(tsv("dev")["client_id"])
    report = {"source": "Common Voice bn (local), train.tsv", "rows_in": len(train)}
    train = train[~train["client_id"].isin(held)]
    report["dropped_speaker_overlap_with_dev_test"] = report["rows_in"] - len(train)
    (args.out / name / "clips").mkdir(parents=True, exist_ok=True)
    rows, dropped_text = [], 0
    for r in train.to_dict("records"):
        t = clean(r["sentence"])
        if t is None:
            dropped_text += 1
            continue
        dst = args.out / name / "clips" / r["path"]
        if not dst.exists():
            shutil.copyfile(root / "clips" / r["path"], dst)
        rows.append({"utt_id": f"cv_train_{Path(r['path']).stem.rsplit('_', 1)[-1]}",
                     "audio_path": f"data/clips/{name}/clips/{r['path']}", "text": t,
                     "group": r["client_id"], "duration": round(float(dur[r["path"]]), 3)})
    report["dropped_text_rules"] = dropped_text
    write_manifest(args.out, name, rows, report)
    zip_folder(args.out, name)


def hf_source(args):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    repo, prefix, text_cols, group_cols = HF[args.cmd]
    name = f"{args.cmd}-train"
    exclude = set(Path(args.exclude_groups).read_text(encoding="utf-8").split()) if args.exclude_groups else set()
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))[: args.max_files]
    report = {"source": f"{repo} {prefix}*", "parquet_files": len(files), "excluded_groups": len(exclude)}
    print(f"{repo}: {len(files)} parquet files")

    # Pass 1: metadata of every row (duration from the audio header), then the random subset.
    meta, n_in, n_excl, n_text = [], 0, 0, 0
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset"))
        cols = pf.schema_arrow.names
        acol, tcol, gcol = pick(cols, ["audio", "audio_filepath"]), pick(cols, text_cols), pick(cols, group_cols)
        if fi == 0:
            print(f"columns {cols}: audio={acol} text={tcol} group={gcol}")
            report["columns"] = {"audio": acol, "text": tcol, "group": gcol}
        row_i = 0
        for batch in pf.iter_batches(batch_size=256, columns=[c for c in (acol, tcol, gcol) if c]):
            for rec in batch.to_pylist():
                i, row_i = row_i, row_i + 1
                n_in += 1
                g = str(rec[gcol]) if gcol else f"{fi}_{i}"
                if g in exclude:
                    n_excl += 1
                    continue
                txt = rec[tcol]
                t = clean(" ".join(map(str, txt)) if isinstance(txt, list) else txt)
                if t is None:
                    n_text += 1
                    continue
                cell = rec[acol]
                d = sf.info(io.BytesIO(cell["bytes"] if isinstance(cell, dict) else cell)).duration
                meta.append({"file": fi, "row": i, "text": t, "group": g, "duration": round(d, 3)})
                if args.limit and len(meta) >= args.limit:
                    break
            if args.limit and len(meta) >= args.limit:
                break
        if args.limit and len(meta) >= args.limit:
            break
    report.update(rows_in=n_in, dropped_speaker_overlap=n_excl, dropped_text_rules=n_text,
                  hours_available=round(sum(m["duration"] for m in meta) / 3600, 2))
    random.Random(SEED).shuffle(meta)
    chosen, total = [], 0.0
    for m in meta:
        if args.max_hours and total >= args.max_hours * 3600:
            break
        if not 0.5 <= m["duration"] <= 30:  # Whisper sees at most 30 s
            continue
        chosen.append(m)
        total += m["duration"]

    # Pass 2: decode the chosen rows, resample to 16 kHz mono, write FLAC.

    want = {}
    for k, m in enumerate(chosen):
        m["utt_id"] = f"{args.cmd}_train_{k:06d}"
        want.setdefault(m["file"], {})[m["row"]] = m
    (args.out / name / "audio").mkdir(parents=True, exist_ok=True)
    for fi, rows_wanted in sorted(want.items()):
        pf = pq.ParquetFile(hf_hub_download(repo, files[fi], repo_type="dataset"))
        acol = report["columns"]["audio"]
        row_i = 0
        for batch in pf.iter_batches(batch_size=256, columns=[acol]):
            for rec in batch.to_pylist():
                m = rows_wanted.get(row_i)
                row_i += 1
                if m is None:
                    continue
                dst = args.out / name / "audio" / f"{m['utt_id']}.flac"
                if not dst.exists():
                    tmp = dst.with_name(dst.stem + ".tmp.flac")
                    sf.write(str(tmp), decode(rec[acol]), 16000, format="FLAC", subtype="PCM_16")
                    tmp.replace(dst)
    rows = [{"utt_id": m["utt_id"], "audio_path": f"data/clips/{name}/audio/{m['utt_id']}.flac", "text": m["text"],
             "group": m["group"], "duration": m["duration"]} for m in chosen]
    write_manifest(args.out, name, rows, report)
    zip_folder(args.out, name)


def merged(args):
    frames = []
    for spec in args.sources:
        src, _, h = spec.rpartition(":") if spec.rpartition(":")[2].replace(".", "").isdigit() else (spec, "", "")
        hours = float(h) if h else args.hours_each
        df = pd.read_csv(src, encoding="utf-8-sig")
        df = df.sample(frac=1.0, random_state=SEED)
        if hours > 0:
            df = df[df["duration"].cumsum() <= hours * 3600]
        if "group" not in df.columns:
            df["group"] = df.get("recording", df["utt_id"])
        frames.append(df[["utt_id", "audio_path", "text", "group", "duration"]].assign(source=Path(src).stem))
        print(f"{src}: {len(df)} rows, {df['duration'].sum() / 3600:.2f} h")
    out = pd.concat(frames, ignore_index=True)
    (args.out / args.name).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / args.name / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"{args.name}: {len(out)} rows, {out['duration'].sum() / 3600:.2f} h")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["cv", "kathbath", "shrutilipi", "merged"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cv_root", default="data/CommonVoiceCorpus/bn")
    p.add_argument("--max_hours", type=float, default=50.0)
    p.add_argument("--exclude_groups", default=None, help="text file, one speaker id per line (Kathbath valid speakers)")
    p.add_argument("--max_files", type=int, default=None, help="testing: only the first N parquet files")
    p.add_argument("--limit", type=int, default=None, help="testing: stop pass 1 after N usable rows")
    p.add_argument("--hours_each", type=float, default=10.0)
    p.add_argument("--from", dest="sources", nargs="*", default=[])
    p.add_argument("--name", default="m5-merged", help="merged: list name (folder under --out)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"cv": cv, "kathbath": hf_source, "shrutilipi": hf_source, "merged": merged}[args.cmd](args)


if __name__ == "__main__":
    main()
