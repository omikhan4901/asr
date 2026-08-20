"""
N-Best Candidate Generation for Bangla ASR Rescoring — BATCHED.

Batches multiple utterances into a single beam-search generate() call for
real GPU parallelism (single-item beam search is memory-bandwidth-bound, not
compute-bound, so batching gives a large throughput win even at "100% GPU
usage" on a single item). Whisper's feature extractor pads every input to a
fixed 30s spectrogram, so batching audio needs no manual padding logic.

Resumable at the UTTERANCE level: each utterance's result is written +
checkpointed immediately after its batch completes, so a crash mid-run only
loses at most one in-flight batch.

FIX (vs. earlier version): num_beams / num_return_sequences are now set
directly on model.generation_config (not just passed as generate() kwargs),
which is what actually makes Whisper's generate() take the beam-search code
path and return a GenerateBeamEncoderDecoderOutput. Sequence scores are now
extracted via compute_transition_scores(), which works correctly regardless
of which output type comes back, instead of assuming outputs.sequences_scores
exists.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import argparse
import traceback
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

SAMPLE_RATE = 16000

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


class NBestGenerator:
    def __init__(
        self,
        model_id: str = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium",
        language: str = "bn",
        num_beams: int = 5,
        max_new_tokens: int = 320,
        debug_first_batch: bool = True,
        diverse_beam_search: bool = False,
        diversity_penalty: float = 1.0,
        sampling_temperature: float = 1.0,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA not available.")

        self.device = torch.device("cuda")
        self.language = language
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self._debug_printed = not debug_first_batch  # print once, on first batch only
        self.diverse_beam_search = diverse_beam_search
        self.diversity_penalty = diversity_penalty
        self.sampling_temperature = sampling_temperature

        print(f"📥 Loading Whisper model in FP16: {model_id}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(self.device)

        # Build the generation config from the ACTUAL checkpoint, not a
        # different base model — mixing configs across checkpoints risks
        # mismatched suppress_tokens / eos handling for the fine-tuned model.
        try:
            base_gen_config = GenerationConfig.from_pretrained(model_id)
        except Exception:
            print("⚠️  No generation_config.json on this checkpoint — "
                  "falling back to openai/whisper-medium's as a base.")
            base_gen_config = GenerationConfig.from_pretrained("openai/whisper-medium")

        self.model.generation_config = base_gen_config

        # Resolve language+task via the tokenizer's own static language
        # tables, NOT via generation_config.language + lang_to_id — some
        # fine-tuned/regional checkpoints ship a generation_config.json
        # without a lang_to_id mapping, which makes the language= kwarg
        # path raise AttributeError. get_decoder_prompt_ids sidesteps that
        # lookup entirely and works regardless of what the checkpoint's
        # config does or doesn't include.
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )
        self.model.generation_config.forced_decoder_ids = forced_decoder_ids

        # --- The actual fix: set beam params ON the config object itself,
        # not only as generate() call kwargs. This is what makes Whisper's
        # generate() reliably take the beam-search path.
        self.model.generation_config.num_beams = self.num_beams
        self.model.generation_config.num_return_sequences = self.num_beams
        self.model.generation_config.early_stopping = True
        self.model.generation_config.output_scores = True
        self.model.generation_config.return_dict_in_generate = True
        # Explicit rather than relying on the default — this directly scales
        # sequences_scores, worth knowing the exact value for your writeup.
        self.model.generation_config.length_penalty = 1.5

        # Plain (greedy/deterministic) beam search frequently collapses to
        # near-identical hypotheses on confident, clearly-articulated audio
        # (narrated reading is close to worst-case for this) — all N beams
        # end up as trivial variations of the same path, which silently
        # makes Oracle WER == 1-best WER and gives every downstream
        # rescoring method nothing to select between.
        #
        # We originally tried Group Beam Search (num_beam_groups>1), but on
        # this transformers version that's been moved out of core into a
        # Hub-hosted "custom generation strategy" that turned out not to be
        # usable (missing custom_generate/generate.py). Rather than keep
        # chasing that version-specific mechanism, we use stochastic beam
        # search instead: do_sample=True combined with num_beams>1 samples
        # from the token distribution at each expansion step instead of
        # always taking the argmax, which directly breaks beam collapse.
        #
        # IMPORTANT — single-call trade-off: do_sample is one flag for the
        # WHOLE call, so there's no way within one generate() to make rank 0
        # deterministic while ranks 1..N-1 are sampled (that requires two
        # separate calls, which costs an extra encoder+decoder pass — see
        # git history for that version if this one isn't fast enough).
        # Temperature is therefore the real dial here: at temperature=1.0,
        # sampling is aggressive enough that even the top-ranked beam is
        # noticeably worse than true deterministic decoding (observed: WER
        # jumped from ~30% to 40.04% on a 10-utterance pilot purely from
        # this contamination, not a real regression). Lowering temperature
        # keeps beam-search scoring dominant at each step, so the top beam
        # stays closer to deterministic quality (baseline stays lower) while
        # still injecting enough randomness for the other beams to diverge.
        # This is a genuine trade-off, not a free lunch: push temperature
        # too low and beams collapse again (Oracle gap -> 0). Tune via
        # --sampling_temperature; default here starts lower (0.7) than our
        # earlier 1.0 specifically to prioritize a believable baseline.
        if self.diverse_beam_search:
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = self.sampling_temperature
            self.model.generation_config.num_beam_groups = 1
        else:
            self.model.generation_config.do_sample = False
            self.model.generation_config.num_beam_groups = 1

        self.model.eval()

        print(f"   model device/dtype: {next(self.model.parameters()).device}, {next(self.model.parameters()).dtype}")
        print(f"   attn implementation: {self.model.config._attn_implementation}")
        print(f"   generation_config.num_beams: {self.model.generation_config.num_beams}")
        if self.diverse_beam_search:
            print(f"   diverse beam search: ON via sampling (do_sample=True, "
                  f"temperature={self.model.generation_config.temperature})")
        else:
            print("   diverse beam search: OFF (plain beam search — watch for Oracle gap ≈ 0)")
        print("✅ Model loaded.")

    def generate_nbest_batch(self, audio_arrays: list[np.ndarray]) -> list[dict]:
        """
        Runs beam search on a BATCH of audio arrays at once.

        Returns a list (one entry per input audio, in the same order) of
        {"candidates": [{"text": str, "logprob": float, "rank": int}, ...]}.

        HF's generate() groups output sequences as:
            [item0_beam0, item0_beam1, ..., item0_beamN-1,
             item1_beam0, item1_beam1, ..., item1_beamN-1, ...]
        i.e. num_return_sequences consecutive rows per input item, in input order.
        """
        batch_size = len(audio_arrays) 
        inputs = self.processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                max_new_tokens=self.max_new_tokens,
                # Explicitly force the generation kwargs here to override HF defaults
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                do_sample=self.diverse_beam_search,
                temperature=self.sampling_temperature if self.diverse_beam_search else 1.0,
                output_scores=True,
                return_dict_in_generate=True, 
            )

        if not self._debug_printed:
            expected_rows = batch_size * self.num_beams
            actual_rows = outputs.sequences.shape[0]
            print(f"🔍 debug: output type={type(outputs).__name__}, "
                  f"sequences.shape={tuple(outputs.sequences.shape)} "
                  f"(expected {expected_rows} rows = batch_size {batch_size} x num_beams {self.num_beams})")
            if actual_rows != expected_rows:
                print("⚠️  WARNING: row count doesn't match batch_size*num_beams — "
                      "beam search may not have engaged. Check transformers version.")
            self._debug_printed = True

        texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        # Now that beam search is confirmed engaging (GenerateBeamEncoderDecoderOutput),
        # sequences_scores is populated directly by HF and is exactly the
        # length-normalized beam score we want — use it. Only fall back to
        # manual computation if it's somehow absent (e.g. a future output-type
        # change), since compute_transition_scores' beam_indices gather has
        # been unreliable on some transformers/CUDA builds here (hit a
        # device-side assert in testing).
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            avg_logprobs = outputs.sequences_scores.float().cpu().numpy().tolist()
        else:
            beam_indices = getattr(outputs, "beam_indices", None)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                beam_indices,
                normalize_logits=True,
            )
            valid_mask = transition_scores > -1e9
            summed = torch.where(valid_mask, transition_scores, torch.zeros_like(transition_scores)).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp(min=1)
            avg_logprobs = (summed / lengths).float().cpu().numpy().tolist()

        results = []
        n = self.num_beams
        for item_idx in range(batch_size):
            start = item_idx * n
            end = start + n
            item_texts = texts[start:end]
            item_scores = avg_logprobs[start:end]
            candidates = [
                {"text": t.strip(), "logprob": s, "rank": i}
                for i, (t, s) in enumerate(zip(item_texts, item_scores))
            ]
            results.append({"candidates": candidates})
        return results


# ── Resumable batch driver ───────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: Path) -> set:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_utt_ids", []))
    return set()


def _save_checkpoint(checkpoint_path: Path, completed_ids: set):
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"completed_utt_ids": sorted(completed_ids)}, f)
    tmp_path.replace(checkpoint_path)


def _chunked(iterable_df: pd.DataFrame, size: int):
    for i in range(0, len(iterable_df), size):
        yield iterable_df.iloc[i:i + size]


def run_nbest_generation(
    split_csv: str,
    output_dir: str,
    model_id: str,
    language: str,
    num_beams: int,
    batch_size: int,
    max_new_tokens: int,
    resume: bool = True,
    limit: int = None,
    diverse_beam_search: bool = False,
    diversity_penalty: float = 1.0,
    sampling_temperature: float = 1.0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_timingtest{limit}" if limit else ""
    nbest_jsonl = output_path / f"nbest_candidates{suffix}.jsonl"
    error_log_path = output_path / f"nbest_errors{suffix}.log"
    checkpoint_path = output_path / f"nbest_checkpoint{suffix}.json"

    if not resume:
        for p in (nbest_jsonl, error_log_path, checkpoint_path):
            p.unlink(missing_ok=True)

    completed_ids = _load_checkpoint(checkpoint_path) if resume else set()

    df = pd.read_csv(split_csv, encoding="utf-8-sig")
    if limit:
        df = df.head(limit)
        print(f"⏱️  TIMING TEST MODE — limited to first {limit} utterances.")

    remaining = df[~df["utt_id"].isin(completed_ids)].reset_index(drop=True)

    print(f"🚀 N-best generation: {len(df)} utterances total, {len(remaining)} remaining.")
    print(f"   batch_size={batch_size}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")
    print(f"   diverse_beam_search={diverse_beam_search}" + (f" (sampling_temperature={sampling_temperature})" if diverse_beam_search else ""))
    if completed_ids:
        print(f"↩️  Resuming — {len(completed_ids)} already done.")

    generator = NBestGenerator(
        model_id=model_id,
        language=language,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        diverse_beam_search=diverse_beam_search,
        diversity_penalty=diversity_penalty,
        sampling_temperature=sampling_temperature,
    )

    n_ok, n_failed = 0, 0
    per_utt_times = []
    run_start = time.time()

    n_batches = (len(remaining) + batch_size - 1) // batch_size

    with open(nbest_jsonl, "a", encoding="utf-8") as f_out:
        for batch_df in tqdm(_chunked(remaining, batch_size), total=n_batches, desc="Generating N-best (batched)"):
            t0 = time.time()

            # Load audio for the whole batch first. If one file is corrupt,
            # drop just that row from the batch rather than failing the batch.
            audio_arrays, valid_rows = [], []
            for _, row in batch_df.iterrows():
                try:
                    arr, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE)
                    # --- NEW: Add 200ms of absolute silence to both ends ---
                    pad_length = int(0.2 * SAMPLE_RATE)
                    arr = np.pad(arr, (pad_length, pad_length), mode='constant')
                    # -------------------------------------------------------
                    audio_arrays.append(arr)
                    valid_rows.append(row)
                except Exception as e:
                    n_failed += 1
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        f_err.write(f"\n--- {row['utt_id']} failed (audio load) ---\n")
                        f_err.write(traceback.format_exc())
                    print(f"❌ {row['utt_id']} failed to load audio ({e}) — skipping.")

            if not audio_arrays:
                continue

            try:
                batch_results = generator.generate_nbest_batch(audio_arrays)
            except Exception as e:
                n_failed += len(valid_rows)
                torch.cuda.empty_cache()
                with open(error_log_path, "a", encoding="utf-8") as f_err:
                    f_err.write(f"\n--- batch starting at {valid_rows[0]['utt_id']} failed (generate) ---\n")
                    f_err.write(traceback.format_exc())
                print(f"❌ Batch generate() failed ({e}) — logged, skipping this batch (will retry next run).")
                continue

            elapsed = time.time() - t0
            per_batch_per_utt = elapsed / len(valid_rows)

            for row, result in zip(valid_rows, batch_results):
                utt_id = row["utt_id"]
                record = {
                    "utt_id": utt_id,
                    "audio_path": row["audio_path"],
                    "ground_truth": row["gt"],
                    "candidates": result["candidates"],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                per_utt_times.append(per_batch_per_utt)

            f_out.flush()

            for row in valid_rows:
                completed_ids.add(row["utt_id"])
            _save_checkpoint(checkpoint_path, completed_ids)
            n_ok += len(valid_rows)

    total_elapsed = time.time() - run_start

    print("\n" + "=" * 60)
    print("🎉 N-BEST GENERATION COMPLETE (this session)")
    print("=" * 60)
    print(f"Newly processed: {n_ok}")
    print(f"Failed (retry):  {n_failed}")
    print(f"Total done:      {len(completed_ids)} / {len(df)}")
    print(f"Output:          {nbest_jsonl}")

    if per_utt_times:
        avg_time = sum(per_utt_times) / len(per_utt_times)
        print("\n--- ⏱️  TIMING REPORT (batched) ---")
        print(f"Session wall time:              {total_elapsed:.1f}s")
        print(f"Effective avg time/utterance:   {avg_time:.2f}s  (batch_size={batch_size})")
        print(f"Utterances/minute:              {60/avg_time:.1f}")

        if limit:
            print(f"\n💡 Extrapolation for full dataset (at this batch_size):")
            for hypothetical_n in [500, 1000, 2000, 5000]:
                est_seconds = avg_time * hypothetical_n
                est_minutes = est_seconds / 60
                print(f"   {hypothetical_n:>5} utterances ≈ {est_minutes:.1f} minutes ({est_seconds/3600:.2f} hours)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, default="data/processed/splits/eval_split.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/nbest")
    parser.add_argument("--model_id", type=str, default="bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    parser.add_argument("--language", type=str, default="bn")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1,
                         help="Number of utterances processed together per generate() call. "
                              "Start at 8, try 16 if VRAM allows — watch nvidia-smi for headroom.")
    parser.add_argument("--max_new_tokens", type=int, default=320,
                         help="Lowered from 444 — a 25s Bangla chunk rarely needs more; "
                              "cuts wasted generation steps across all beams.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--diverse_beam_search", action="store_true",
                         help="Use diverse beam search (num_beam_groups=num_beams, one beam per group) "
                              "instead of plain beam search. Plain beam search often collapses to "
                              "near-identical candidates on confident/clearly-articulated audio, which "
                              "makes Oracle WER == 1-best WER and gives rescoring nothing to work with. "
                              "Turn this on if evaluation.py reports an oracle gap near 0.")
    parser.add_argument("--diversity_penalty", type=float, default=1.0,
                         help="Unused (kept for compatibility) — group beam search was replaced by "
                              "sampling-based diversity; see --sampling_temperature instead.")
    parser.add_argument("--sampling_temperature", type=float, default=0.7,
                         help="Sampling temperature used with --diverse_beam_search (do_sample=True + "
                              "num_beams>1), single fast call. Lower = top beam (your baseline) stays "
                              "closer to true deterministic quality but candidates diverge less; "
                              "higher = more diversity but a noisier, worse baseline (1.0 measured "
                              "40.04%% WER vs. deterministic's ~30%% on a 10-utt pilot). Start at 0.7; "
                              "raise toward 1.0 only if the Oracle gap stays near 0, and watch the "
                              "baseline number when you do.")
    args = parser.parse_args()

    run_nbest_generation(
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        model_id=args.model_id,
        language=args.language,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        diverse_beam_search=args.diverse_beam_search,
        diversity_penalty=args.diversity_penalty,
        sampling_temperature=args.sampling_temperature,
    )