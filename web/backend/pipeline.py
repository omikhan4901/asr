"""
End-to-end single-video processing pipeline for the web demo.

Reuses the exact generation config confirmed as the project's winning
setup (num_beams=12, sampling_temperature=0.7, diverse beam search via
do_sample) and the probability-weighted MBR formula from
src/mbr_rescore.py, applied per-chunk to one uploaded video instead of a
sharded batch job.

DEPLOYMENT NOTE: this uses the actual Silero VAD model (same as Phase 1's
preprocess_chunk_align.py), not a lighter substitute — the deployed demo
should reflect the real pipeline, not an approximation of it.
"""

import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import librosa
import torch
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000
MODEL_ID = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"

# Confirmed winning config from the project's calibration sweep
NUM_BEAMS = 12
SAMPLING_TEMPERATURE = 0.7
LENGTH_PENALTY = 1.5
MAX_NEW_TOKENS = 320
MAX_CHUNK_SECONDS = 25.0
MIN_CHUNK_SECONDS = 0.5


def extract_audio(video_path: str, out_wav_path: str):
    """ffmpeg: any video/audio container -> 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-vn",
        out_wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-800:]}")


def chunk_audio(audio: np.ndarray, sr: int, vad_model) -> list[dict]:
    """
    Silero VAD speech-interval detection (same model as Phase 1), then
    greedily packs intervals into bounded chunks matching Phase 1's bounds
    (0.5s - 25.0s), cutting only at detected silence.
    """
    audio_tensor = torch.from_numpy(audio).float()
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=sr, return_seconds=True
    )  # [{"start": float_sec, "end": float_sec}, ...]

    if not speech_timestamps:
        return [{"start": 0.0, "end": len(audio) / sr, "audio": audio}]

    chunks = []
    cur_start, cur_end = speech_timestamps[0]["start"], speech_timestamps[0]["end"]
    for seg in speech_timestamps[1:]:
        proposed_len = seg["end"] - cur_start
        if proposed_len <= MAX_CHUNK_SECONDS:
            cur_end = seg["end"]
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
    chunks.append((cur_start, cur_end))

    result = []
    for s, e in chunks:
        dur = e - s
        if dur < MIN_CHUNK_SECONDS:
            continue
        start_sample, end_sample = int(s * sr), int(e * sr)
        result.append({"start": s, "end": e, "audio": audio[start_sample:end_sample]})
    return result


def normalize_bangla_text(text: str) -> str:
    import re
    text = str(text).strip()
    text = re.sub(r"[।,\-.?!:;\"'()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def mbr_select(candidates: list[dict]) -> dict:
    """Same probability-weighted MBR formula as src/mbr_rescore.py."""
    texts = [normalize_bangla_text(c["text"]) for c in candidates]
    logprobs = np.array([c["logprob"] for c in candidates])
    weights = softmax(logprobs)

    n = len(texts)
    risks = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                loss = jiwer.wer(texts[j], texts[i])
            except Exception:
                loss = 1.0
            risks[i] += weights[j] * loss

    best_idx = int(np.argmin(risks))
    return candidates[best_idx]


class SubtitleGenerator:
    """Loads the model once; call .process_video() per request."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        self.device = torch.device("cuda")

        self.processor = WhisperProcessor.from_pretrained(MODEL_ID)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa",
        ).to(self.device)

        # Load the BASE checkpoint's generation_config directly, not the
        # fine-tuned model's own — bengaliAI's checkpoint ships an
        # old-style generation_config.json that loads without error but is
        # incompatible with the language=/task= generate() API (see:
        # github.com/huggingface/transformers/issues/25084). This matches
        # the fix already proven working in inference_batched.py.
        base_gen_config = GenerationConfig.from_pretrained("openai/whisper-medium")
        self.model.generation_config = base_gen_config
        self.model.generation_config.forced_decoder_ids = None

        self.model.eval()

        # Same VAD model as Phase 1 — loaded once per container, reused
        # across all requests this container handles.
        self.vad_model = load_silero_vad()

    def _generate_nbest(self, audio: np.ndarray) -> list[dict]:
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                max_new_tokens=MAX_NEW_TOKENS,
                language="bn",
                task="transcribe",
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_BEAMS,
                early_stopping=True,
                length_penalty=LENGTH_PENALTY,
                do_sample=True,
                temperature=SAMPLING_TEMPERATURE,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        # Same safety fallback as modal_inference.py / inference.py: with
        # do_sample=True, Whisper's generate() can return a plain
        # GenerateEncoderDecoderOutput (no sequences_scores attribute) even
        # with num_beams>1, instead of the beam-search-shaped output.
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            logprobs = outputs.sequences_scores.float().cpu().numpy().tolist()
        else:
            beam_indices = getattr(outputs, "beam_indices", None)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, beam_indices, normalize_logits=True,
            )
            valid_mask = transition_scores > -1e9
            summed = torch.where(valid_mask, transition_scores, torch.zeros_like(transition_scores)).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp(min=1)
            logprobs = (summed / lengths).float().cpu().numpy().tolist()

        return [{"text": t.strip(), "logprob": lp, "rank": i} for i, (t, lp) in enumerate(zip(texts, logprobs))]

    def process_video(self, video_path: str, progress_cb=None) -> dict:
        """
        progress_cb(stage: str, current: int|None, total: int|None) is
        called at each pipeline stage so the caller (app.py) can update the
        job-status store for the frontend's async polling.
        """
        def report(stage, current=None, total=None):
            if progress_cb:
                progress_cb(stage, current, total)

        report("audio")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "audio.wav")
            extract_audio(video_path, wav_path)
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        total_duration = len(audio) / SAMPLE_RATE

        report("chunk")
        chunks = chunk_audio(audio, SAMPLE_RATE, self.vad_model)
        if not chunks:
            raise RuntimeError("No speech detected in this video.")

        report("asr", 0, len(chunks))
        for i, chunk in enumerate(chunks):
            chunk["candidates"] = self._generate_nbest(chunk["audio"])
            report("asr", i + 1, len(chunks))

        report("rescore", 0, len(chunks))
        for i, chunk in enumerate(chunks):
            selected = mbr_select(chunk["candidates"])
            chunk["text"] = selected["text"]
            report("rescore", i + 1, len(chunks))

        report("srt")
        from srt_utils import chunks_to_srt, srt_preview
        srt_text = chunks_to_srt(chunks)

        return {
            "srt_text": srt_text,
            "srt_preview": srt_preview(srt_text),
            "n_segments": len([c for c in chunks if c.get("text", "").strip()]),
            "duration_seconds": total_duration,
        }