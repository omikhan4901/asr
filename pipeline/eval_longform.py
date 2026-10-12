#!/usr/bin/env python3
"""
Whole-recording evaluation (Phase 5): no reference-guided cuts. Each raw recording is cut by voice activity
detection only (Silero VAD, the web demo's chunker, chunks <= its max length), every chunk is transcribed,
the texts are joined in order, and WER is computed against the recording's full transcript
(data/raw/train/annotation/<recording>.txt, normalized like every other number). This is real long-form
recognition: the forced-alignment clip split never touches the audio the model sees.

  PYTHONPATH=src python pipeline/eval_longform.py --recordings_dir data/longform --model models/m5_full \
      --name m5_full_t4 --out_dir results/longform
Recordings: <dir>/audio/<rec>.flac|.wav and <dir>/annotation/<rec>.txt (runs/phase5_upload_longform.sh builds
the 22 DL Sprint eval recordings). Output: <out_dir>/<name>/recordings.jsonl (one line per recording, resumable)
and metrics.json (corpus WER over recordings with a 95% bootstrap CI over recordings, speed).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_zero_shot import build  # noqa: E402

from bangla_asr.common.audio import load_audio  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.preprocessing.vad import SileroVADChunker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings_dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("results/longform"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_seconds", type=float, default=None, help="tests only: first N seconds of each recording")
    ap.add_argument("--limit", type=int, default=None, help="tests only: first N recordings")
    args = ap.parse_args()

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "recordings.jsonl"
    done = {json.loads(l)["recording"] for l in res_path.open(encoding="utf-8")} if res_path.exists() else set()
    recs = sorted(p.stem for p in (args.recordings_dir / "annotation").glob("*.txt"))[: args.limit]
    todo = [r for r in recs if r not in done]
    print(f"{len(recs)} recordings, {len(done)} done, {len(todo)} to go -> {out}", flush=True)
    if todo:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = build(args.model, "whisper", device, args.adapter, args.num_beams)
        vad = SileroVADChunker()
        for rec in todo:
            audio_file = next(p for ext in (".flac", ".wav") if (p := args.recordings_dir / "audio" / f"{rec}{ext}").exists())
            wav, _ = load_audio(audio_file, 16000)
            if args.max_seconds:
                wav = wav[: int(args.max_seconds * 16000)]
            t0 = time.time()
            chunks = vad.segment_audio(wav)
            texts = []
            for i in range(0, len(chunks), args.batch_size):
                texts += model.transcribe([c["audio"] for c in chunks[i:i + args.batch_size]])
            elapsed = time.time() - t0
            ref = normalize_bangla_text((args.recordings_dir / "annotation" / f"{rec}.txt").read_text(encoding="utf-8")).split()
            hyp = normalize_bangla_text(" ".join(texts)).split()
            row = {"recording": rec, "audio_seconds": round(len(wav) / 16000, 1), "chunks": len(chunks),
                   "ref_words": len(ref), "hyp_words": len(hyp), "edits": Levenshtein.distance(ref, hyp),
                   "seconds": round(elapsed, 1), "hyp": " ".join(hyp)}
            with res_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{rec}: WER {100 * row['edits'] / max(1, row['ref_words']):.2f}% ({len(chunks)} chunks, "
                  f"{row['audio_seconds'] / 60:.1f} min audio in {elapsed / 60:.1f} min)", flush=True)

    rows = [json.loads(l) for l in res_path.open(encoding="utf-8")]
    e = np.array([r["edits"] for r in rows], float)
    n = np.array([r["ref_words"] for r in rows], float)
    rng = np.random.default_rng(0)
    boots = [e[i].sum() / n[i].sum() for i in (rng.integers(0, len(rows), len(rows)) for _ in range(2000))]
    metrics = {"model": args.model, "num_beams": args.num_beams, "recordings": len(rows),
               "wer": float(e.sum() / n.sum()), "wer_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               "audio_hours": round(sum(r["audio_seconds"] for r in rows) / 3600, 3),
               "rtf": round(sum(r["seconds"] for r in rows) / max(1e-9, sum(r["audio_seconds"] for r in rows)), 4),
               "test_mode": bool(args.max_seconds or args.limit)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"whole-recording WER {100 * metrics['wer']:.2f}% [{100 * metrics['wer_ci95'][0]:.2f}, "
          f"{100 * metrics['wer_ci95'][1]:.2f}] over {len(rows)} recordings, RTF {metrics['rtf']}", flush=True)


if __name__ == "__main__":
    main()
