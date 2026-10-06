#!/usr/bin/env python3
"""
Transcribe a test manifest with one model and score it (E2: zero-shot baselines).

Works for Whisper-family models and CTC models (wav2vec2 etc.) on Hugging Face.
Manifests are the CSVs written by pipeline/02_split.py (long-form chunks with
start/end) or pipeline/data/prepare_hf.py (one WAV per utterance).

Outputs in --out_dir/<set name>/<model slug>/:
  predictions.jsonl   utt_id, group, ref, hyp (resumable: finished utt_ids are skipped)
  metrics.json        corpus WER/CER, 95% cluster-bootstrap CI, runtime, real-time factor

Example:
  PYTHONPATH=src ./Scripts/python.exe pipeline/eval_zero_shot.py \
      --manifest data/splits_fa/eval.csv --model bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_row_audio
from bangla_asr.common.stats import corpus_rates, edit_counts


class CTCModel:
    def __init__(self, model_id, device):
        from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

        self.device = device
        # Plain processor (no bundled n-gram LM): zero-shot rows are acoustic-only greedy.
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCTC.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def transcribe(self, audios):
        inp = self.processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        logits = self.model(inp.input_values.to(self.device, self.dtype),
                            attention_mask=inp.get("attention_mask", None).to(self.device)
                            if inp.get("attention_mask", None) is not None else None).logits
        ids = logits.argmax(-1)
        return [t.strip() for t in self.processor.batch_decode(ids)]


class WhisperModel:
    def __init__(self, model_id, device):
        from bangla_asr.config import WhisperConfig
        from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel

        self.m = WhisperASRModel(WhisperConfig(model_id=model_id, num_beams=1, diverse_beam_search=False),
                                 device=str(device))

    def transcribe(self, audios):
        return self.m.transcribe_batch(audios, sampling_rate=16000)


def build(model_id, family, device):
    if family == "auto":
        from transformers import AutoConfig
        arch = (AutoConfig.from_pretrained(model_id).architectures or [""])[0].lower()
        family = "whisper" if "whisper" in arch else "ctc"
    return (WhisperModel if family == "whisper" else CTCModel)(model_id, device), family


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--family", choices=["auto", "whisper", "ctc"], default="auto")
    p.add_argument("--set_name", default=None, help="Defaults to the manifest's parent/stem, e.g. splits_fa-eval")
    p.add_argument("--out_dir", default="results/zeroshot")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    man = Path(args.manifest)
    set_name = args.set_name or f"{man.parent.name}-{man.stem}"
    out = Path(args.out_dir, set_name, args.model.replace("/", "__"))
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"

    df = pd.read_csv(man, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)
    ref_col = "text" if "text" in df.columns else "gt"
    group_col = "recording" if "recording" in df.columns else ("group" if "group" in df.columns else "utt_id")
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["utt_id"] for l in pred_path.open(encoding="utf-8") if l.strip()}
    todo = df[~df["utt_id"].isin(done)]
    # Sort by duration so batches pad less (much faster).
    if "duration" in todo.columns:
        todo = todo.sort_values("duration", ascending=False)
    print(f"{set_name} | {args.model}: {len(df)} utterances, {len(done)} done, {len(todo)} to go -> {out}")

    if len(todo):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, family = build(args.model, args.family, device)
        t0, audio_s = time.time(), 0.0
        with pred_path.open("a", encoding="utf-8") as f:
            rows = todo.to_dict("records")
            for i in tqdm(range(0, len(rows), args.batch_size), desc="transcribing"):
                batch = rows[i:i + args.batch_size]
                audios = [load_row_audio(r) for r in batch]
                audio_s += sum(len(a) for a in audios) / 16000
                try:
                    hyps = model.transcribe(audios)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    hyps = [h for a in audios for h in model.transcribe([a])]
                for r, h in zip(batch, hyps):
                    f.write(json.dumps({"utt_id": r["utt_id"], "group": str(r[group_col]),
                                        "ref": str(r[ref_col]), "hyp": h}, ensure_ascii=False) + "\n")
                f.flush()
        elapsed = time.time() - t0
    else:
        family, elapsed, audio_s = args.family, None, None

    preds = [json.loads(l) for l in pred_path.open(encoding="utf-8") if l.strip()]
    ids = set(df["utt_id"])
    preds = [r for r in preds if r["utt_id"] in ids]
    scored = [dict(edit_counts(r["hyp"], r["ref"]), group=r["group"]) for r in preds]
    metrics = corpus_rates(scored, group_key="group")
    metrics.update({"set": set_name, "model": args.model, "family": family, "manifest": str(man),
                    "normalizer": "bangla_asr.common.text.normalize_bangla_text", "decoding": "greedy"})
    if elapsed:
        metrics.update({"last_run_seconds": round(elapsed, 1), "last_run_audio_hours": round(audio_s / 3600, 3),
                        "last_run_rtf": round(elapsed / max(1e-9, audio_s), 4)})
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lo, hi = metrics.get("wer_ci95", [float("nan")] * 2)
    print(f"WER {100 * metrics['wer']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]  CER {100 * metrics['cer']:.2f}%  "
          f"({metrics['n_utterances']} utt, {metrics.get('n_groups')} groups)")


if __name__ == "__main__":
    main()
