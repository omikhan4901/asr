"""Save N random chunks of a split as WAV + reference text, for listening checks.

Usage (from the repo root):
    PYTHONPATH=src ./Scripts/python.exe analysis/export_spotcheck.py --split eval --n 20

Writes spotcheck/<split>/NN_<utt_id>.wav and a matching .txt, plus index.csv
with an empty "matches_audio" column to fill in while listening.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, "src")
from bangla_asr.common.audio import load_segment  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="eval")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="spotcheck")
    args = p.parse_args()

    df = pd.read_csv(f"data/splits_fa/{args.split}.csv", encoding="utf-8-sig")
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    out = Path(args.out, args.split)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in sample.iterrows():
        name = f"{i + 1:02d}_{r['utt_id']}"
        sf.write(out / f"{name}.wav", load_segment(r["audio_path"], r["start"], r["end"]), 16000)
        (out / f"{name}.txt").write_text(str(r["text"]), encoding="utf-8")
        rows.append({"file": name, "recording": r["recording"], "start": r["start"], "end": r["end"],
                     "mean_score": r["mean_score"], "matches_audio": "", "note": ""})
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False, encoding="utf-8-sig")
    print(f"{len(rows)} chunks -> {out}  (fill in matches_audio: yes / partly / no)")


if __name__ == "__main__":
    main()
