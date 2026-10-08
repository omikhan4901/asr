#!/usr/bin/env python3
"""
Re-score an existing N-best file with exact candidate log-probabilities.

N-best files written before 2026-10-08 store a "logprob" per candidate that is
not the candidate's probability: current transformers returns neither
sequences_scores nor beam_indices for Whisper beam search, and the fallback
read per-step scores from the wrong beams. Candidate texts, ranks, oracle and
agreement are unaffected; MBR (which weights candidates by these scores) is.

This script keeps everything and replaces "logprob" with the mean per-token
log-probability of the candidate text under the model (teacher forcing,
T = 1, same function as new N-best runs: WhisperASRModel.score_texts). The
old value is kept as "logprob_generate". Writes a NEW file (--output_jsonl);
resumable like the generator.

Example (~30 min for the 4,234-chunk eval 12-best on an RTX 3060):
  PYTHONPATH=src ./Scripts/python.exe pipeline/rescore_nbest.py \
      --input_jsonl data/nbest_fa/eval_nbest.jsonl --manifest data/splits_fa/eval.csv \
      --output_jsonl data/nbest_fa/eval_sampled.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_row_audio, pad_silence
from bangla_asr.common.provenance import RunRecord
from bangla_asr.common.safe_io import read_jsonl, repair_jsonl
from bangla_asr.config import AudioConfig, WhisperConfig
from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--manifest", required=True, help="the manifest the N-best was generated from (start/end slices)")
    p.add_argument("--model_id", default=None)
    args = p.parse_args()

    src, dst = Path(args.input_jsonl), Path(args.output_jsonl)
    if src.resolve() == dst.resolve():
        raise SystemExit("--output_jsonl must differ from --input_jsonl (the original is kept)")
    repair_jsonl(dst)
    done = {r["utt_id"] for r in read_jsonl(dst)}
    recs = [r for r in read_jsonl(src) if r["utt_id"] not in done]
    rows = {r["utt_id"]: r for r in pd.read_csv(args.manifest, encoding="utf-8-sig").to_dict("records")}
    missing = [r["utt_id"] for r in recs if r["utt_id"] not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} utterances not in {args.manifest} (first: {missing[0]})")
    print(f"{src}: {len(recs)} to score, {len(done)} already done -> {dst}")

    cfg = WhisperConfig(**({"model_id": args.model_id} if args.model_id else {}))
    audio_cfg = AudioConfig()
    with RunRecord(dst.with_suffix(".run_info.json"), inputs=[src, args.manifest]) as rec:
        if recs:
            model = WhisperASRModel(cfg)
            with open(dst, "a", encoding="utf-8") as f:
                for r in tqdm(recs, desc="rescoring", mininterval=10):
                    audio = pad_silence(load_row_audio(rows[r["utt_id"]]), audio_cfg.silence_pad_seconds, audio_cfg.sample_rate)
                    cands = r["candidates"]
                    try:
                        scored = model.score_texts(audio, [c["text"] for c in cands])
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        scored = [model.score_texts(audio, [c["text"]])[0] for c in cands]
                    for c, s in zip(cands, scored):
                        c["logprob_generate"] = c.get("logprob")
                        c["logprob"] = s["mean_logprob"]
                        c["sum_logprob"], c["n_tokens"] = s["sum_logprob"], s["n_tokens"]
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
        rec.outputs = [dst]
        rec.extra = {"model_id": cfg.model_id, "score": "mean token log-probability, teacher forcing, T=1"}
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
