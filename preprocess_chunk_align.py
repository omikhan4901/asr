"""
Bangla ASR Preprocessing: VAD-Based Audio Chunking & Ground-Truth Fuzzy Alignment
Optimized with explicit CUDA device allocation and FP16 half-precision.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import argparse
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from rapidfuzz import fuzz
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import json
import os
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SAMPLE_RATE = 16000

# Minimum fuzzy-match score (0-100) required to accept a chunk's ground-truth
# alignment and advance the sequential corpus pointer. Chunks scoring below
# this are still kept in the manifest (for auditing) but dropped from the
# final aligned dataset, and the pointer is NOT advanced on their behalf.
MIN_ALIGNMENT_SCORE = 50


# ── Text Normalization Utilities ─────────────────────────────────────────────

def clean_text_for_alignment(text: str) -> str:
    """Strip Bengali punctuation and whitespace for robust fuzzy matching."""
    return re.sub(r"[।,\-.?!:;\"'()]", "", str(text)).strip()


# ── Voice Activity Detection ─────────────────────────────────────────────────

class VadChunker:
    """
    Uses Silero VAD to find speech segments, then greedily packs consecutive
    segments into chunks of at most `chunk_max_duration_s`, splitting only at
    silence boundaries. This avoids cutting audio mid-word, which otherwise
    produces garbled ASR hypotheses at chunk edges and corrupts alignment.
    """

    def __init__(self, chunk_max_duration_s: float, min_chunk_duration_s: float):
        self.chunk_max_duration_s = chunk_max_duration_s
        self.min_chunk_duration_s = min_chunk_duration_s

        print("📥 Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.get_speech_timestamps = utils[0]

    def get_chunks(self, audio_array: np.ndarray) -> List[Dict]:
        """Returns a list of {start_sample, end_sample} spans to transcribe."""
        speech_timestamps = self.get_speech_timestamps(
            torch.from_numpy(audio_array),
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
        )

        if not speech_timestamps:
            return []

        max_samples = int(self.chunk_max_duration_s * SAMPLE_RATE)
        min_samples = int(self.min_chunk_duration_s * SAMPLE_RATE)

        chunks = []
        cur_start = speech_timestamps[0]["start"]
        cur_end = speech_timestamps[0]["end"]

        for seg in speech_timestamps[1:]:
            # If adding this speech segment would exceed the max chunk length,
            # close out the current chunk and start a new one at a silence boundary.
            if seg["end"] - cur_start > max_samples:
                chunks.append({"start_sample": cur_start, "end_sample": cur_end})
                cur_start = seg["start"]
                cur_end = seg["end"]
            else:
                cur_end = seg["end"]

        chunks.append({"start_sample": cur_start, "end_sample": cur_end})

        return [c for c in chunks if c["end_sample"] - c["start_sample"] >= min_samples]


# ── Audio Processing & ASR Hypothesis Generation ─────────────────────────────

class ChunkingPreprocessor:
    def __init__(
        self,
        model_id: str = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium",
        chunk_max_duration_s: float = 25.0,
        min_chunk_duration_s: float = 0.5,
        language: str = "bn",
    ):
        # Explicitly check and configure for CUDA GPU
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA is not available! PyTorch cannot detect your GPU. "
                "Ensure you have installed the CUDA-enabled version of PyTorch."
            )

        self.device = torch.device("cuda")
        print(f"🖥️ Using GPU: {torch.cuda.get_device_name(0)}")

        self.language = language
        self.vad = VadChunker(chunk_max_duration_s, min_chunk_duration_s)

        print(f"📥 Loading Whisper model in FP16: {model_id}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Force decoding into Bangla transcription (not translation, not
        # auto-detected language) so short/quiet/noisy chunks can't silently
        # decode into the wrong language and poison fuzzy alignment.
        # NOTE: newer `transformers` versions deprecated passing
        # `forced_decoder_ids` directly to generate(); language/task are now
        # passed as generate() kwargs instead (see transcribe_chunk below).
        #
        # The bengaliAI checkpoint ships an outdated generation_config.json
        # (predates lang_to_id / is_multilingual support), which makes
        # generate(language=..., task=...) raise a ValueError. Since this
        # checkpoint is a fine-tune of openai/whisper-medium, we can safely
        # pull a modern generation_config from the base model instead.
        from transformers import GenerationConfig

        base_gen_config = GenerationConfig.from_pretrained("openai/whisper-medium")
        self.model.generation_config = base_gen_config
        self.model.generation_config.language = self.language
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

        self.model.eval()
        print("✅ Whisper model loaded successfully onto GPU.")

    def transcribe_chunk(self, audio_array: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Transcribe an isolated 1-D audio chunk using the GPU. (Unbatched — kept for
        manual debugging; the pipeline itself uses transcribe_batch below.)"""
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                max_new_tokens=444,
                language=self.language,
                task="transcribe",
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def transcribe_batch(self, chunk_audios: List[np.ndarray], sr: int = SAMPLE_RATE) -> List[str]:
        """Transcribe a list of 1-D audio chunks in a single batched GPU forward pass."""
        # The processor extracts and pads the features to a uniform tensor shape in the
        # batch. return_attention_mask=True is required here — WhisperFeatureExtractor
        # does not return an attention_mask by default, and without one the model can't
        # distinguish real audio from padding in the shorter chunks of a batch.
        inputs = self.processor(
            chunk_audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                max_new_tokens=444,
                language=self.language,
                task="transcribe",
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def chunk_and_transcribe(
        self,
        audio_path: str,
        output_audio_dir: str,
        batch_size: int = 16,
        max_chunk_duration: float = None,
    ) -> List[Dict]:
        """
        Splits long-form audio into VAD-bounded chunks, batches them into groups of
        `batch_size`, transcribes them in parallel on the GPU, and exports the WAVs.

        VAD spans longer than `max_chunk_duration` are further split into fixed-length
        sub-chunks. This is a safety net: VadChunker already caps chunks at
        chunk_max_duration_s, but if a single continuous speech segment (no detected
        pause) exceeds that cap, it's emitted as one oversized chunk (see
        VadChunker.get_chunks). Whisper's feature extractor silently truncates any
        input beyond 30s (3000 mel frames), so without this split, such a span would
        lose audio and produce an incomplete transcript with no warning.

        Defaults to the VAD's own configured cap (self.vad.chunk_max_duration_s) rather
        than a hardcoded value, so it always matches whatever --chunk_size was passed.
        """
        os.makedirs(output_audio_dir, exist_ok=True)
        stem_name = Path(audio_path).stem

        if max_chunk_duration is None:
            max_chunk_duration = self.vad.chunk_max_duration_s

        # Wrap the loader in a try-except block to catch corrupted audio files
        try:
            audio_array, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"\n  ❌ CORRUPT FILE DETECTED ({stem_name}): {e}")
            print(f"  ↳ Skipping {stem_name} and continuing to the next file...")
            return [] # Return empty chunks list so the pipeline moves on

        vad_spans = self.vad.get_chunks(audio_array)
        if not vad_spans:
            print(f"  ⚠️  No speech detected in {stem_name}.")
            return []

        max_chunk_samples = int(max_chunk_duration * sr)

        # 1. First pass: collect valid raw audio segments, splitting any span
        #    that exceeds max_chunk_duration into contiguous sub-spans.
        valid_spans = []
        for span in vad_spans:
            start_sample, end_sample = span["start_sample"], span["end_sample"]
            span_len = end_sample - start_sample

            if span_len <= max_chunk_samples:
                sub_bounds = [(start_sample, end_sample)]
            else:
                sub_bounds = []
                s = start_sample
                while s < end_sample:
                    e = min(s + max_chunk_samples, end_sample)
                    sub_bounds.append((s, e))
                    s = e

            for sub_start, sub_end in sub_bounds:
                chunk_audio = audio_array[sub_start:sub_end]
                duration = len(chunk_audio) / sr
                valid_spans.append({
                    "start_sample": sub_start,
                    "end_sample": sub_end,
                    "chunk_audio": chunk_audio,
                    "duration": duration,
                })

        chunks = []

        # 2. Second pass: process in batches of `batch_size` (e.g., 8 or 16)
        for i in range(0, len(valid_spans), batch_size):
            batch_spans = valid_spans[i : i + batch_size]
            batch_audios = [item["chunk_audio"] for item in batch_spans]

            # Run batched GPU inference
            texts = self.transcribe_batch(batch_audios, sr)

            # Process results and write to disk
            for local_idx, (span_info, text) in enumerate(zip(batch_spans, texts)):
                if not text.strip():
                    continue

                chunk_id = i + local_idx
                start_sample = span_info["start_sample"]
                end_sample = span_info["end_sample"]
                chunk_audio = span_info["chunk_audio"]
                duration = span_info["duration"]

                out_wav_path = os.path.join(output_audio_dir, f"{stem_name}_{chunk_id:03d}.wav")
                sf.write(out_wav_path, chunk_audio, sr)

                chunks.append({
                    "chunk_id": chunk_id,
                    "audio_path": out_wav_path,
                    "text": text.strip(),
                    "start": round(start_sample / sr, 3),
                    "end": round(end_sample / sr, 3),
                    "duration": round(duration, 3),
                })

        return chunks


# ── Ground-Truth Sequential Fuzzy Alignment ──────────────────────────────────

def align_chunks_sequential(file_df: pd.DataFrame, annotation_file_path: str) -> pd.DataFrame:
    """
    Maps each ASR chunk hypothesis to the best-matching span of the ground-truth
    text using a sliding sequential pointer with RapidFuzz ratio scoring.

    Low-confidence matches (score < MIN_ALIGNMENT_SCORE) are recorded but do NOT
    advance the pointer, so a single bad ASR hypothesis can't cascade and corrupt
    alignment for every subsequent chunk in the file.
    """
    with open(annotation_file_path, "r", encoding="utf-8") as f:
        corpus = f.read().strip()

    corpus_words = clean_text_for_alignment(corpus).split()
    gt_texts = []
    scores = []
    corpus_pos = 0

    for _, row in file_df.iterrows():
        asr_clean = clean_text_for_alignment(row["text"])
        asr_words = asr_clean.split()

        if not asr_words or corpus_pos >= len(corpus_words):
            gt_texts.append("")
            scores.append(0)
            continue

        num_asr_words = len(asr_words)
        best_score = 0
        best_length = num_asr_words
        best_start = corpus_pos

        # Search within a sliding window [-5, +10] words around the current pointer
        for start_offset in range(-5, 10):
            start = max(0, corpus_pos + start_offset)
            for length_offset in range(-3, 5):
                length = max(min(5, num_asr_words), num_asr_words + length_offset)
                end = min(start + length, len(corpus_words))
                if start >= end:
                    continue

                candidate = " ".join(corpus_words[start:end])
                score = fuzz.ratio(asr_clean, candidate)
                if score > best_score:
                    best_score = score
                    best_length = end - start
                    best_start = start

        gt_end = min(best_start + best_length, len(corpus_words))
        gt_text = " ".join(corpus_words[best_start:gt_end])

        gt_texts.append(gt_text)
        scores.append(best_score)

        # Only trust the pointer advance if the match was good enough.
        # Otherwise leave corpus_pos where it was so the next chunk still
        # searches from a sane position instead of drifting off a bad match.
        if best_score >= MIN_ALIGNMENT_SCORE:
            corpus_pos = gt_end

    result_df = file_df.copy()
    result_df["gt"] = gt_texts
    result_df["align_score"] = scores
    return result_df


# ── Execution Pipeline ───────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: Path) -> set:
    """Returns the set of file_nums that have already been fully processed."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed_file_nums", []))
    return set()


def _save_checkpoint(checkpoint_path: Path, completed_file_nums: set):
    # Write to a temp file then rename — avoids a corrupted checkpoint if the
    # process dies mid-write.
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"completed_file_nums": sorted(completed_file_nums)}, f)
    tmp_path.replace(checkpoint_path)


def _append_df_to_csv(df: pd.DataFrame, csv_path: Path):
    """Appends rows to a CSV, writing the header only if the file is new."""
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")

def _write_stats_file(
    output_path: Path,
    full_manifest_csv: Path,
    final_aligned_csv: Path,
    completed_file_nums: set,
    total_files_requested: int,
    min_alignment_score: int,
    n_ok_this_session: int,
    n_skipped_missing_this_session: int,
    n_failed_this_session: int,
):
    stats_path = output_path / "dataset_stats.txt"

    df_full = pd.read_csv(full_manifest_csv, encoding="utf-8-sig") if full_manifest_csv.exists() else pd.DataFrame()
    df_aligned = pd.read_csv(final_aligned_csv, encoding="utf-8-sig") if final_aligned_csv.exists() else pd.DataFrame()

    lines = []
    lines.append("=" * 60)
    lines.append("BANGLA ASR PREPROCESSING — DATASET STATISTICS")
    lines.append("=" * 60)
    lines.append(f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("")

    # --- Coverage ---
    lines.append("-- File Coverage --")
    lines.append(f"Files requested:        {total_files_requested}")
    lines.append(f"Files completed (total): {len(completed_file_nums)}")
    lines.append(f"Files remaining:         {total_files_requested - len(completed_file_nums)}")
    lines.append(f"  (this session — ok: {n_ok_this_session}, "
                  f"missing: {n_skipped_missing_this_session}, failed: {n_failed_this_session})")
    lines.append("")

    # --- Chunk-level stats ---
    lines.append("-- Chunk Extraction --")
    lines.append(f"Total raw chunks extracted:      {len(df_full)}")
    if not df_full.empty and "file_num" in df_full.columns:
        per_file_counts = df_full.groupby("file_num").size()
        lines.append(f"Avg chunks per file:              {per_file_counts.mean():.2f}")
        lines.append(f"Min / Max chunks in a single file: {per_file_counts.min()} / {per_file_counts.max()}")
    lines.append("")

    # --- Alignment quality ---
    lines.append("-- Alignment Quality --")
    if not df_full.empty and "align_score" in df_full.columns:
        n_below_threshold = (df_full["align_score"] < min_alignment_score).sum()
        n_empty_gt = (df_full["gt"].fillna("").str.len() == 0).sum()
        lines.append(f"Alignment score threshold (τ):    {min_alignment_score}")
        lines.append(f"Mean alignment score:              {df_full['align_score'].mean():.2f}")
        lines.append(f"Median alignment score:            {df_full['align_score'].median():.2f}")
        lines.append(f"Min / Max alignment score:         {df_full['align_score'].min():.2f} / {df_full['align_score'].max():.2f}")
        lines.append(f"Chunks below threshold (dropped):  {n_below_threshold} "
                      f"({100 * n_below_threshold / len(df_full):.1f}%)")
        lines.append(f"Chunks with empty ground truth:    {n_empty_gt}")
    else:
        lines.append("No alignment data available yet.")
    lines.append("")

    # --- Final dataset ---
    lines.append("-- Final Aligned Dataset (used for training/eval) --")
    lines.append(f"Total valid samples:               {len(df_aligned)}")
    if len(df_full) > 0:
        retention_pct = 100 * len(df_aligned) / len(df_full)
        lines.append(f"Retention rate (aligned/raw):       {retention_pct:.1f}%")
    lines.append("")

    # --- Audio duration stats: RAW (all extracted chunks) ---
    if not df_full.empty and "duration" in df_full.columns:
        total_hours = df_full["duration"].sum() / 3600
        lines.append("-- Audio Duration (Raw / All Extracted Chunks) --")
        lines.append(f"Total raw audio duration:          {total_hours:.2f} hours")
        lines.append(f"Mean chunk duration:                {df_full['duration'].mean():.2f} s")
        lines.append(f"Min / Max chunk duration:           {df_full['duration'].min():.2f} s / {df_full['duration'].max():.2f} s")
        lines.append("")

    # --- Audio duration stats: FINAL (retained/aligned chunks only) ---
    # aligned_chunks.csv only carries [audio_path, gt, align_score] — join back
    # to the full manifest on audio_path to recover duration/file_num for the
    # subset that actually survives into the training/eval set.
    if not df_aligned.empty and not df_full.empty and "duration" in df_full.columns:
        df_final = df_aligned.merge(
            df_full[["audio_path", "duration", "file_num"]],
            on="audio_path",
            how="left",
        )
        if df_final["duration"].notna().any():
            final_hours = df_final["duration"].sum() / 3600
            lines.append("-- Audio Duration (Final Usable Training Set) --")
            lines.append(f"Total usable audio duration:       {final_hours:.2f} hours")
            lines.append(f"Mean chunk duration:                {df_final['duration'].mean():.2f} s")
            lines.append(f"Min / Max chunk duration:           {df_final['duration'].min():.2f} s / {df_final['duration'].max():.2f} s")
            n_final_files = df_final["file_num"].nunique()
            lines.append(f"Source files represented:           {n_final_files}")
            lines.append("")

    # --- ASR hypothesis vs. ground-truth word-length comparison ---
    # Surfaces systematic length mismatches (e.g. Whisper hallucinating extra
    # words, or under-transcribing) that raw align_score alone won't show.
    if not df_full.empty and "text" in df_full.columns and "gt" in df_full.columns:
        text_wc = df_full["text"].fillna("").str.split().str.len()
        gt_wc = df_full["gt"].fillna("").str.split().str.len()
        # Only compare rows where alignment actually produced a non-empty gt,
        # otherwise the "0-length gt" rows would just drag the stats down.
        mask = gt_wc > 0
        lines.append("-- ASR Hypothesis vs. Ground-Truth Word Counts --")
        lines.append(f"Mean ASR hypothesis length:          {text_wc.mean():.2f} words")
        if mask.any():
            lines.append(f"Mean ground-truth span length:       {gt_wc[mask].mean():.2f} words")
            mean_diff = (text_wc[mask] - gt_wc[mask]).mean()
            lines.append(f"Mean (ASR len - GT len):             {mean_diff:+.2f} words")
        lines.append("")

    lines.append("=" * 60)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📊 Dataset stats written to: {stats_path}")
    return stats_path

def run_pipeline(
    audio_dir: str,
    annotation_dir: str,
    output_dir: str,
    start_file: int,
    num_files: int,
    chunk_size: float,
    model_id: str,
    language: str,
    min_alignment_score: int,
    resume: bool = True,
):
    global MIN_ALIGNMENT_SCORE
    MIN_ALIGNMENT_SCORE = min_alignment_score

    output_path = Path(output_dir)
    output_audio_path = output_path / "audio"
    output_audio_path.mkdir(parents=True, exist_ok=True)

    full_manifest_csv = output_path / "chunks_manifest_with_scores.csv"
    final_aligned_csv = output_path / "aligned_chunks.csv"
    error_log_path = output_path / "errors.log"
    checkpoint_path = output_path / "checkpoint.json"

    # Fresh run: wipe any stale outputs so appends don't mix old + new data.
    if not resume:
        for p in (full_manifest_csv, final_aligned_csv, error_log_path, checkpoint_path):
            p.unlink(missing_ok=True)

    completed_file_nums = _load_checkpoint(checkpoint_path) if resume else set()

    preprocessor = ChunkingPreprocessor(
        model_id=model_id,
        chunk_max_duration_s=chunk_size,
        language=language,
    )

    end_file = start_file + num_files
    file_range = list(range(start_file, end_file))
    remaining = [f for f in file_range if f not in completed_file_nums]

    print(f"\n🚀 Processing audio range: train_{start_file:03d}.wav → train_{end_file - 1:03d}.wav")
    if completed_file_nums:
        print(f"↩️  Resuming — {len(completed_file_nums)} files already done, {len(remaining)} remaining.")

    n_ok, n_skipped_missing, n_failed = 0, 0, 0

    for file_num in tqdm(remaining, desc="Processing files"):
        audio_filename = f"train_{file_num:03d}.wav"
        text_filename = f"train_{file_num:03d}.txt"

        audio_path = os.path.join(audio_dir, audio_filename)
        text_path = os.path.join(annotation_dir, text_filename)

        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            print(f"⚠️  Missing file for {file_num:03d} — skipping (not marked complete, will retry next run).")
            n_skipped_missing += 1
            continue

        try:
            # 1. Chunk + transcribe this ONE file.
            chunks = preprocessor.chunk_and_transcribe(
                audio_path=audio_path,
                output_audio_dir=str(output_audio_path),
            )
            if not chunks:
                # Still mark complete — an empty result is a valid outcome, not a crash.
                completed_file_nums.add(file_num)
                _save_checkpoint(checkpoint_path, completed_file_nums)
                continue

            for chunk in chunks:
                chunk["file_num"] = file_num
                chunk["text_path"] = text_path

            file_df = pd.DataFrame(chunks)

            # 2. Align THIS file's chunks immediately (not batched at the end).
            aligned_file_df = align_chunks_sequential(file_df, text_path)

            # 3. Persist immediately — this is what makes it crash-safe.
            _append_df_to_csv(aligned_file_df, full_manifest_csv)

            good_rows = aligned_file_df[
                (aligned_file_df["gt"].str.len() > 0)
                & (aligned_file_df["align_score"] >= MIN_ALIGNMENT_SCORE)
            ][["audio_path", "gt", "align_score"]]
            _append_df_to_csv(good_rows, final_aligned_csv)

            # 4. Only mark complete AFTER both writes succeed.
            completed_file_nums.add(file_num)
            _save_checkpoint(checkpoint_path, completed_file_nums)
            n_ok += 1

        except Exception as e:
            n_failed += 1
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- train_{file_num:03d} failed ---\n")
                f.write(traceback.format_exc())
            print(f"❌ train_{file_num:03d} failed ({e}) — logged, continuing to next file.")
            continue  # do NOT mark complete — next run will retry this file

    _write_stats_file(
        output_path=output_path,
        full_manifest_csv=full_manifest_csv,
        final_aligned_csv=final_aligned_csv,
        completed_file_nums=completed_file_nums,
        total_files_requested=len(file_range),
        min_alignment_score=MIN_ALIGNMENT_SCORE,
        n_ok_this_session=n_ok,
        n_skipped_missing_this_session=n_skipped_missing,
        n_failed_this_session=n_failed,
    )

    print("\n" + "=" * 60)
    print("🎉 PIPELINE RUN COMPLETE (this session)")
    print("=" * 60)
    print(f"Newly processed:     {n_ok}")
    print(f"Skipped (missing):   {n_skipped_missing}")
    print(f"Failed (will retry): {n_failed}")
    print(f"Total done overall:  {len(completed_file_nums)} / {len(file_range)}")
    print(f"Full manifest:       {full_manifest_csv}")
    print(f"Aligned dataset:     {final_aligned_csv}")
    print(f"Checkpoint:          {checkpoint_path}")
    if n_failed:
        print(f"⚠️  See {error_log_path} for failure details — just rerun the same command to retry.")
    print("=" * 60)

# ── Command-Line Interface ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess long-form Bangla audio into VAD-aligned chunks using GPU."
    )
    parser.add_argument("--audio_dir", type=str, default="data/raw/train/audio", help="Path to 1-hour WAV files")
    parser.add_argument("--annotation_dir", type=str, default="data/raw/train/annotation", help="Path to 1-hour TXT files")
    parser.add_argument("--output_dir", type=str, default="data/processed/chunks", help="Destination directory")
    parser.add_argument("--start_file", type=int, default=1, help="Starting file index (e.g. 1 for train_001.wav)")
    parser.add_argument("--num_files", type=int, default=2, help="Number of files to process")
    parser.add_argument("--chunk_size", type=float, default=25.0, help="Max chunk length in seconds")
    parser.add_argument("--model_id", type=str, default="bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    parser.add_argument("--language", type=str, default="bn", help="Whisper decoding language code")
    parser.add_argument("--min_alignment_score", type=int, default=50, help="Min fuzzy match score (0-100) to accept a chunk")

    args = parser.parse_args()

    run_pipeline(
        audio_dir=args.audio_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        start_file=args.start_file,
        num_files=args.num_files,
        chunk_size=args.chunk_size,
        model_id=args.model_id,
        language=args.language,
        min_alignment_score=args.min_alignment_score,
    )