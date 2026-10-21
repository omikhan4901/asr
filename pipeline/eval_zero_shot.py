#!/usr/bin/env python3
"""
Transcribe a test manifest with one model and score it (E2: zero-shot baselines).

Works for Whisper-family models and CTC models (wav2vec2 etc.) on Hugging Face.
Manifests are the CSVs written by pipeline/02_split.py (clips cut from long recordings, with
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
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # silence per-batch generate() warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.common.audio import load_row_audio
from bangla_asr.common.safe_io import atomic_write_text, read_jsonl, repair_jsonl
from bangla_asr.common.provenance import RunRecord
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
    def __init__(self, model_id, device, adapter=None, num_beams=1):
        from bangla_asr.config import WhisperConfig

        self.num_beams = num_beams
        from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel

        self.m = WhisperASRModel(WhisperConfig(model_id=model_id, num_beams=1, diverse_beam_search=False,
                                               adapter_path=adapter), device=str(device))

    def transcribe(self, audios):
        return self.m.transcribe_batch(audios, sampling_rate=16000, num_beams=self.num_beams)


class IndicConformerModel:
    """AI4Bharat IndicConformer (ONNX, trust_remote_code; gated on HF). One utterance per call, language 'bn';
    decoding 'rnnt' or 'ctc' (both greedy in the released code)."""

    def __init__(self, model_id, device, decoding):
        from transformers import AutoModel

        self.decoding = decoding
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)  # picks the GPU itself (ONNX + TorchScript)

    @torch.inference_mode()
    def transcribe(self, audios):
        return [str(self.model(torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0), "bn", self.decoding)).strip()
                for a in audios]


class CT2WhisperModel:
    """A Whisper model converted to CTranslate2 (faster-whisper), e.g. a DL Sprint 4.0 finalist's release.
    Greedy: beam 1, temperature 0 only (no fallback), no timestamps, no VAD, no previous-text conditioning."""

    def __init__(self, model_id, device):
        from faster_whisper import WhisperModel as FW

        cuda = device.type == "cuda"
        self.m = FW(model_id, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")

    def transcribe(self, audios):
        out = []
        for a in audios:
            segs, _ = self.m.transcribe(np.asarray(a, dtype=np.float32), language="bn", task="transcribe", beam_size=1,
                                        temperature=0.0, without_timestamps=True, condition_on_previous_text=False,
                                        vad_filter=False)
            out.append(" ".join(seg.text.strip() for seg in segs).strip())
        return out


def build(model_id, family, device, adapter=None, num_beams=1):
    if family == "auto":
        if "indic-conformer" in model_id:
            family = "indicconformer-rnnt"
        elif model_id.endswith("-ct2"):
            family = "ct2"
        else:
            from transformers import AutoConfig
            arch = (AutoConfig.from_pretrained(model_id).architectures or [""])[0].lower()
            family = "whisper" if "whisper" in arch else "ctc"
    if family == "whisper":
        return WhisperModel(model_id, device, adapter, num_beams), family
    if family.startswith("indicconformer-"):
        return IndicConformerModel(model_id, device, family.split("-")[1]), family
    if family == "ct2":
        return CT2WhisperModel(model_id, device), family
    return CTCModel(model_id, device), family


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--family", choices=["auto", "whisper", "ctc", "indicconformer-rnnt", "indicconformer-ctc", "ct2"],
                   default="auto")
    p.add_argument("--num_beams", type=int, default=1, help="Whisper only: 1 = greedy (every reported row), >1 = beam search")
    p.add_argument("--adapter", default=None, help="LoRA adapter folder (pipeline/11_finetune.py final/) on top of --model")
    p.add_argument("--name", default=None, help="Output folder name; default: the model id, or <run folder> for an adapter")
    p.add_argument("--set_name", default=None, help="Defaults to the manifest's parent/stem, e.g. splits_fa-eval")
    p.add_argument("--out_dir", default="results/zeroshot")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    man = Path(args.manifest)
    set_name = args.set_name or f"{man.parent.name}-{man.stem}"
    name = args.name or (Path(args.adapter).resolve().parent.name if args.adapter else args.model.replace("/", "__"))
    out = Path(args.out_dir, set_name, name)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    adapter_files = sorted(Path(args.adapter).glob("adapter_*")) if args.adapter else []
    with RunRecord(out / "run_info.json", inputs=[man, *adapter_files]) as rec:

        df = pd.read_csv(man, encoding="utf-8-sig")
        if args.limit:
            df = df.head(args.limit)
        ref_col = "text" if "text" in df.columns else "gt"
        group_col = "recording" if "recording" in df.columns else ("group" if "group" in df.columns else "utt_id")
        repair_jsonl(pred_path)  # drop a half-written line left by a power cut
        done = {r["utt_id"] for r in read_jsonl(pred_path)}
        todo = df[~df["utt_id"].isin(done)]
        # Sort by duration so batches pad less (much faster).
        if "duration" in todo.columns:
            todo = todo.sort_values("duration", ascending=False)
        print(f"{set_name} | {args.model}: {len(df)} utterances, {len(done)} done, {len(todo)} to go -> {out}")

        if len(todo):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, family = build(args.model, args.family, device, args.adapter, args.num_beams)
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

        preds = read_jsonl(pred_path)
        ids = set(df["utt_id"])
        preds = [r for r in preds if r["utt_id"] in ids]
        scored = [dict(edit_counts(r["hyp"], r["ref"]), group=r["group"]) for r in preds]
        metrics = corpus_rates(scored, group_key="group")
        metrics.update({"set": set_name, "model": args.model, "adapter": args.adapter, "num_beams": args.num_beams, "family": family, "manifest": str(man),
                        "normalizer": "bangla_asr.common.text.normalize_bangla_text", "decoding": "greedy"})
        if elapsed:
            metrics.update({"last_run_seconds": round(elapsed, 1), "last_run_audio_hours": round(audio_s / 3600, 3),
                            "last_run_rtf": round(elapsed / max(1e-9, audio_s), 4)})
        atomic_write_text(out / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
        rec.outputs = [pred_path, out / "metrics.json"]
        lo, hi = metrics.get("wer_ci95", [float("nan")] * 2)
        print(f"WER {100 * metrics['wer']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]  CER {100 * metrics['cer']:.2f}%  "
              f"({metrics['n_utterances']} utt, {metrics.get('n_groups')} groups)")


if __name__ == "__main__":
    main()
