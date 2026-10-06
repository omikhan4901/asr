#!/usr/bin/env python3
"""
Step 1 (499B): forced-alignment segmentation of long recordings.

For every recording with a transcript, aligns the full transcript to the full
audio with MMS_FA, then cuts word-boundary chunks (<= 25 s). Replaces the
499A VAD + Whisper-rough-transcript + fuzzy-match pipeline (01_preprocess.py).

Outputs (in --output_dir):
  words/<recording>.json   word-level timestamps + scores (for analysis)
  chunks/<recording>.json  chunks with reference text + alignment scores
  segments.csv             all chunks of all recordings (rebuilt every run)
  segment_log.csv          per-recording timing and coverage

Audio is NOT copied: chunks store (recording audio path, start, end).
Resumable: recordings whose chunks/<recording>.json exists are skipped
(use --overwrite to redo them).

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/01_segment.py --limit 1
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_audio
from bangla_asr.config import ForcedAlignConfig
from bangla_asr.preprocessing.forced_align import MMSForcedAligner, chunk_words


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio_dir", default="data/raw/train/audio")
    p.add_argument("--annotation_dir", default="data/raw/train/annotation")
    p.add_argument("--output_dir", default="data/processed_fa")
    p.add_argument("--limit", type=int, default=None, help="Process at most this many recordings (for testing)")
    p.add_argument("--only", nargs="*", default=None, help="Recording stems to process, e.g. train_001")
    p.add_argument("--overwrite", action="store_true", help="Re-segment recordings that already have output")
    args = p.parse_args()

    cfg = ForcedAlignConfig()
    out = Path(args.output_dir)
    (out / "words").mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    if args.only:
        audio_files = [a for a in audio_files if a.stem in set(args.only)]
    todo = [a for a in audio_files
            if (Path(args.annotation_dir) / f"{a.stem}.txt").exists()
            and (args.overwrite or not (out / "chunks" / f"{a.stem}.json").exists())]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(audio_files)} recordings found, {len(todo)} to process -> {out}")

    aligner = MMSForcedAligner(cfg) if todo else None
    log_rows = []
    log_path = out / "segment_log.csv"
    if args.overwrite and log_path.exists() and not args.only:
        log_path.unlink()

    for i, wav in enumerate(todo, 1):
        t0 = time.time()
        text = (Path(args.annotation_dir) / f"{wav.stem}.txt").read_text(encoding="utf-8-sig")
        words = text.split()
        try:
            audio, sr = load_audio(wav, target_sr=aligner.sample_rate)
        except Exception as exc:  # e.g. a 0-byte file in the release
            print(f"[{i}/{len(todo)}] {wav.stem}: SKIPPED ({exc})")
            pd.DataFrame([{"recording": wav.stem, "error": str(exc)}]).to_csv(
                out / "skipped.csv", mode="a", header=not (out / "skipped.csv").exists(), index=False)
            continue
        dur = len(audio) / sr

        aligned = aligner.align(audio, words)
        chunks = chunk_words(aligned, cfg, total_duration=dur)
        for c in chunks:
            c["recording"] = wav.stem
            c["audio_path"] = str(wav).replace("\\", "/")

        (out / "words" / f"{wav.stem}.json").write_text(
            json.dumps([asdict(w) for w in aligned], ensure_ascii=False), encoding="utf-8")
        (out / "chunks" / f"{wav.stem}.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

        n_al = sum(w.aligned for w in aligned)
        covered = sum(c["duration"] for c in chunks)
        row = {
            "recording": wav.stem,
            "audio_min": round(dur / 60, 2),
            "words": len(words),
            "aligned_words": n_al,
            "chunks": len(chunks),
            "chunk_audio_min": round(covered / 60, 2),
            "median_word_score": round(float(pd.Series([w.score for w in aligned if w.aligned]).median()), 4) if n_al else None,
            "anchors": aligner.last_stats.get("anchors"),
            "pieces": aligner.last_stats.get("pieces"),
            "seconds": round(time.time() - t0, 1),
        }
        log_rows.append(row)
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        print(f"[{i}/{len(todo)}] {wav.stem}: {dur/60:.1f} min, {len(words)} words, "
              f"{len(chunks)} chunks, median word score {row['median_word_score']}, {row['seconds']} s")

    # Rebuild the combined manifest from every per-recording chunk file.
    all_chunks = []
    for f in sorted((out / "chunks").glob("*.json")):
        all_chunks.extend(json.loads(f.read_text(encoding="utf-8")))
    if all_chunks:
        df = pd.DataFrame(all_chunks)
        df.insert(0, "utt_id", df["recording"] + "_" + df.groupby("recording").cumcount().map("{:04d}".format))
        df.to_csv(out / "segments.csv", index=False, encoding="utf-8-sig")
        print(f"segments.csv: {len(df)} chunks, {df['duration'].sum()/3600:.1f} h, "
              f"from {df['recording'].nunique()} recordings")


if __name__ == "__main__":
    main()
