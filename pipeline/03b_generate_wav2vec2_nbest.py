#!/usr/bin/env python3
"""
Step 3b: CTC N-best generation with Wav2Vec2 + pyctcdecode.

Does not replace pipeline/03_generate_nbest.py (the published Whisper path).
Requires: pip install pyctcdecode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_audio
from bangla_asr.config import Wav2Vec2Config
from bangla_asr.models.ctc.wav2vec2_model import Wav2Vec2Model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wav2Vec2 CTC N-best candidate pools.")
    parser.add_argument("--split_csv", type=str, default="data/processed/splits/eval_split.csv")
    parser.add_argument("--output_jsonl", type=str, default="data/processed/nbest/eval_wav2vec2_nbest.jsonl")
    parser.add_argument("--model_id", type=str, default="arijitx/wav2vec2-xls-r-300m-bengali")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--kenlm_model_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)
    cfg = Wav2Vec2Config(model_id=args.model_id, beam_width=args.n, kenlm_model_path=args.kenlm_model_path)
    model = Wav2Vec2Model(cfg)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Wav2Vec2 N-best"):
            try:
                audio, _ = load_audio(row["audio_path"], target_sr=16000)
            except Exception as exc:
                print(f"skip {row.get('utt_id')}: {exc}")
                continue
            greedy = model.transcribe(audio)
            nbest = model.transcribe_nbest(audio, n=args.n)
            record = {
                "utt_id": row["utt_id"],
                "audio_path": row["audio_path"],
                "ground_truth": row.get("gt", ""),
                "model_family": model.model_family,
                "model_id": cfg.model_id,
                "greedy_text": greedy.text,
                "candidates": [h.to_candidate_dict() for h in nbest],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
