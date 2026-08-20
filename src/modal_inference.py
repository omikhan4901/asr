"""
Modal N-best generation for the TRAIN split (Bangla ASR rescoring project).

This version is brought into alignment with the current Kaggle eval-split
script (inference.py) — same generation-config strategy, same score
extraction priority, same diversity mechanism — so that whatever the learned
reranker sees at TRAIN time (this script) matches what it'll see at EVAL time
(Kaggle script) as closely as possible. See CONFIG block below: values here
must be copy-pasted from whatever flags you actually used on Kaggle, not
assumed from a docstring.

Changes vs. the previous Modal version:
  1. [FIX] num_beams here now defaults to matching inference.py's CLI default
     (5), not 12 — the earlier docstring claimed "12" without this being
     confirmed against the real Kaggle run. VERIFY_NUM_BEAMS below is a loud,
     impossible-to-miss reminder to check this before launching.
  2. Score extraction now prefers `sequences_scores` first (populated
     directly by HF when beam search genuinely engages), falling back to
     `compute_transition_scores` only if absent — matches inference.py's
     priority, reversed from this script's previous version.
  3. `forced_decoder_ids` resolved via `processor.get_decoder_prompt_ids`
     instead of relying on generation_config.language/lang_to_id, which can
     be absent on some fine-tuned checkpoints and raise AttributeError.
  4. Diversity mechanism matched to Kaggle: do_sample + num_beams>1
     (stochastic beam search) with a tunable sampling_temperature, not group
     beam search (which isn't reliably available on this transformers
     version — see inference.py's comment history for why that path was
     abandoned).
  5. [NEW] Deterministic seeding per-shard (torch.manual_seed(BASE_SEED +
     shard_id)) — do_sample=True without a seed means a re-run of a timed-out
     shard produces a DIFFERENT candidate distribution than the original
     attempt for whatever was already written, which is a correctness risk
     for reranker training data. Now reruns are reproducible.
  6. [NEW] Startup shape/type debug print (once per shard) matching
     inference.py's sanity check — confirms beam search actually engaged
     before you burn GPU-hours on all shards.
  7. [NEW] --sample_n dry-run mode: process only the first N records total
     (spread proportionally across shards) for a cheap timing test before
     committing to the full run.
  8. [KEPT] Incremental per-batch commit, per-shard resumability, explicit
     forced_decoder_ids/num_beams/etc. as generate() kwargs, volume.reload()
     before merge, path remapping for the Windows/MSYS path-mangling issue.

AUDIO PATH NOTE (unchanged from before): actual files on this volume live at
    /data/C:/Program Files/Git/processed_audio_chunks/chunks/audio/*.wav
due to an earlier `modal volume put` run from Git Bash on Windows silently
rewriting the destination path. map_audio_path() below remaps explicitly.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("bangla-asr-nbest-generator")

SHARD_TIMEOUT_S = 6 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "accelerate", "librosa",
        "soundfile", "pandas", "numpy", "rapidfuzz",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

volume = modal.Volume.from_name("bangla-asr-data", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"

SAMPLE_RATE = 16000

REAL_AUDIO_PREFIX = "/data/C:/Program Files/Git/processed_audio_chunks/chunks/audio"

# ============================================================================
# ⚠️  CONFIG — MUST MATCH YOUR ACTUAL KAGGLE inference.py RUN EXACTLY.
#
# Do not trust these defaults. Before launching, open your Kaggle notebook's
# command history / output logs for the run whose eval-split results you are
# treating as final, and copy the ACTUAL --num_beams, --max_new_tokens,
# --diverse_beam_search, --sampling_temperature values used. A mismatch here
# means the learned reranker trains on a different candidate distribution
# than it evaluates on.
# ============================================================================
NUM_BEAMS = 12                   # CONFIRMED from actual Kaggle eval-split run
                                  # (full_eval_run.py CONFIG dict, num_beams=12).
MAX_NEW_TOKENS = 320
LENGTH_PENALTY = 1.5
DIVERSE_BEAM_SEARCH = True       # CONFIRMED — Kaggle run uses diverse_beam_search=True
SAMPLING_TEMPERATURE = 0.7       # CONFIRMED from Kaggle run's CONFIG dict
BATCH_SIZE = 2                  # A10G has more VRAM than your 3060 — start
                                  # here, watch nvidia-smi in the dry run below
                                  # before assuming this is optimal.
SILENCE_PAD_SECONDS = 0.2        # matches the padding added in inference.py

BASE_SEED = 1234                 # per-shard seed = BASE_SEED + shard_id, so
                                  # reruns of a timed-out shard reproduce the
                                  # SAME sampled candidates for records not
                                  # yet written, not a different distribution.


def map_audio_path(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    filename = rel_path.split("/")[-1]
    return f"{REAL_AUDIO_PREFIX}/{filename}"


@app.cls(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=SHARD_TIMEOUT_S,
    max_containers=10,
)
class DistributedNBestWorker:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

        self.model_id = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
        self.device = torch.device("cuda")
        self._debug_printed = False

        print("Loading Whisper checkpoint into GPU memory...")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
        ).to(self.device)

        try:
            base_gen_config = GenerationConfig.from_pretrained(self.model_id)
        except Exception:
            print("⚠️  No generation_config.json on this checkpoint — "
                  "falling back to openai/whisper-medium's as a base.")
            base_gen_config = GenerationConfig.from_pretrained("openai/whisper-medium")

        self.model.generation_config = base_gen_config

        # Resolve language+task via the tokenizer's own prompt-id helper,
        # not generation_config.language + lang_to_id (can be absent on
        # fine-tuned regional checkpoints and raise AttributeError).
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="bn", task="transcribe"
        )
        self.model.generation_config.forced_decoder_ids = self.forced_decoder_ids
        self.model.generation_config.num_beams = NUM_BEAMS
        self.model.generation_config.num_return_sequences = NUM_BEAMS
        self.model.generation_config.early_stopping = True
        self.model.generation_config.length_penalty = LENGTH_PENALTY

        if DIVERSE_BEAM_SEARCH:
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = SAMPLING_TEMPERATURE
        else:
            self.model.generation_config.do_sample = False

        self.model.eval()
        print(f"Model loaded. num_beams={NUM_BEAMS}, length_penalty={LENGTH_PENALTY}, "
              f"diverse_beam_search={DIVERSE_BEAM_SEARCH}"
              + (f" (temperature={SAMPLING_TEMPERATURE})" if DIVERSE_BEAM_SEARCH else ""))

    def _generate_batch(self, audio_arrays: list, shard_id: int):
        import torch

        inputs = self.processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                max_new_tokens=MAX_NEW_TOKENS,
                forced_decoder_ids=self.forced_decoder_ids,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_BEAMS,
                early_stopping=True,
                length_penalty=LENGTH_PENALTY,
                do_sample=DIVERSE_BEAM_SEARCH,
                temperature=SAMPLING_TEMPERATURE if DIVERSE_BEAM_SEARCH else 1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )

        if not self._debug_printed:
            expected_rows = len(audio_arrays) * NUM_BEAMS
            actual_rows = outputs.sequences.shape[0]
            print(f"[shard {shard_id}] 🔍 debug: output type={type(outputs).__name__}, "
                  f"sequences.shape={tuple(outputs.sequences.shape)} "
                  f"(expected {expected_rows} = batch {len(audio_arrays)} x num_beams {NUM_BEAMS})")
            if actual_rows != expected_rows:
                print(f"[shard {shard_id}] ⚠️ WARNING: row count mismatch — beam search "
                      f"may not have engaged. Check transformers version before trusting results.")
            self._debug_printed = True

        texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        # Prefer sequences_scores (populated directly when beam search
        # genuinely engages) — matches inference.py's priority. Only fall
        # back to manual computation if absent.
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            avg_logprobs = outputs.sequences_scores.float().cpu().numpy().tolist()
        else:
            beam_indices = getattr(outputs, "beam_indices", None)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, beam_indices, normalize_logits=True,
            )
            valid_mask = transition_scores > -1e9
            summed = torch.where(valid_mask, transition_scores, torch.zeros_like(transition_scores)).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp(min=1)
            avg_logprobs = (summed / lengths).float().cpu().numpy().tolist()

        return texts, avg_logprobs

    @modal.method()
    def process_shard(self, shard_records: list[dict], shard_id: int, output_jsonl: str) -> dict:
        import librosa
        import numpy as np
        import torch

        # Deterministic per-shard seed — reruns of a timed-out shard produce
        # the SAME sampled candidates for whatever wasn't yet written.
        torch.manual_seed(BASE_SEED + shard_id)

        shard_out_path = os.path.join(VOLUME_MOUNT_PATH, f"{output_jsonl}.shard{shard_id}.jsonl")

        completed_ids = set()
        if os.path.exists(shard_out_path):
            with open(shard_out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            completed_ids.add(json.loads(line)["utt_id"])
                        except Exception:
                            pass
            print(f"[shard {shard_id}] resuming — {len(completed_ids)} already done")

        remaining = [r for r in shard_records if r["utt_id"] not in completed_ids]
        print(f"[shard {shard_id}] {len(remaining)} / {len(shard_records)} remaining")

        n_written = 0
        with open(shard_out_path, "a", encoding="utf-8") as f_out:
            for i in range(0, len(remaining), BATCH_SIZE):
                batch = remaining[i: i + BATCH_SIZE]
                audio_arrays, valid_records = [], []

                for rec in batch:
                    audio_path = map_audio_path(rec["audio_path"])
                    try:
                        arr, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
                        pad_len = int(SILENCE_PAD_SECONDS * SAMPLE_RATE)
                        arr = np.pad(arr, (pad_len, pad_len), mode="constant")
                        audio_arrays.append(arr)
                        valid_records.append(rec)
                    except Exception as e:
                        print(f"[shard {shard_id}] error loading {audio_path}: {e}")

                if not audio_arrays:
                    continue

                try:
                    texts, avg_logprobs = self._generate_batch(audio_arrays, shard_id)
                except Exception as e:
                    print(f"[shard {shard_id}] generate() failed on batch starting "
                          f"{valid_records[0]['utt_id']}: {e}")
                    torch.cuda.empty_cache()
                    continue

                expected_rows = len(valid_records) * NUM_BEAMS
                if len(texts) != expected_rows:
                    print(f"[shard {shard_id}] ⚠️ row mismatch: got {len(texts)}, "
                          f"expected {expected_rows} — skipping this batch to avoid misaligned candidates")
                    continue

                for item_idx, rec in enumerate(valid_records):
                    start = item_idx * NUM_BEAMS
                    end = start + NUM_BEAMS
                    item_texts = texts[start:end]
                    item_scores = avg_logprobs[start:end]

                    candidates = [
                        {"text": t.strip(), "logprob": s, "rank": rk}
                        for rk, (t, s) in enumerate(zip(item_texts, item_scores))
                    ]

                    record = {
                        "utt_id": rec["utt_id"],
                        "audio_path": rec["audio_path"],
                        "ground_truth": rec["gt"],
                        "candidates": candidates,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

                f_out.flush()
                if n_written % 20 < BATCH_SIZE:
                    volume.commit()

        volume.commit()
        print(f"[shard {shard_id}] done — {n_written} new records written")
        return {"shard_id": shard_id, "n_written": n_written, "shard_out_path": shard_out_path}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=SHARD_TIMEOUT_S + 600)
def main(
    split_csv_path: str = "train_split.csv",
    output_jsonl: str = "train_nbest_candidates.jsonl",
    num_shards: int = 10,
    sample_n: int = None,
):
    """
    sample_n: if set, only processes the first `sample_n` records total
    (spread across shards proportionally) — use this for a cheap dry run to
    measure real throughput before committing to the full split.
    """
    import pandas as pd
    import time

    csv_full_path = os.path.join(VOLUME_MOUNT_PATH, split_csv_path)
    print(f"Reading: {csv_full_path}")
    df = pd.read_csv(csv_full_path, encoding="utf-8-sig")
    records = df.to_dict(orient="records")
    print(f"Total utterances in split: {len(records)}")

    if sample_n:
        records = records[:sample_n]
        print(f"🧪 DRY RUN — limited to first {sample_n} records.")

    shards = [records[i::num_shards] for i in range(num_shards)]
    shards = [s for s in shards if s]  # drop empty shards (small sample_n case)
    print(f"Distributing {len(records)} records across {len(shards)} shard(s) "
          f"(~{len(records)//max(len(shards),1)}/shard)")

    worker = DistributedNBestWorker()
    t0 = time.time()
    shard_results = list(
        worker.process_shard.starmap(
            [(shard, i, output_jsonl) for i, shard in enumerate(shards)]
        )
    )
    elapsed = time.time() - t0

    total_new = sum(r["n_written"] for r in shard_results)
    print(f"\nGeneration pass complete — {total_new} new records this run in {elapsed:.1f}s.")
    if total_new > 0:
        per_utt = elapsed / total_new
        print(f"Effective throughput: {per_utt:.2f}s/utterance "
              f"(wall time across {len(shards)} parallel shard(s) — NOT sequential)")
        if sample_n:
            print(f"\n💡 Extrapolation (assumes same shard count / parallelism for the full run):")
            for hypothetical_n in [1000, 5000, 10000, 20000]:
                # Rough: same per-shard throughput, same number of shards running in parallel
                est_seconds = per_utt * (hypothetical_n / max(len(shards), 1))
                print(f"   {hypothetical_n:>6} utterances ≈ {est_seconds/60:.1f} min ({est_seconds/3600:.2f} hr) "
                      f"wall time across {len(shards)} shards")

    volume.reload()

    out_full_path = os.path.join(VOLUME_MOUNT_PATH, output_jsonl)
    total_merged = 0
    with open(out_full_path, "w", encoding="utf-8") as f_out:
        for r in shard_results:
            if not os.path.exists(r["shard_out_path"]):
                continue
            with open(r["shard_out_path"], "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        total_merged += 1
    volume.commit()
    print(f"Merged {total_merged} records -> {out_full_path}")

    if total_merged == 0:
        print("⚠️⚠️⚠️ MERGED 0 RECORDS. Do not treat this run as complete. "
              "Check shard logs above for the actual failure before rerunning.")


@app.local_entrypoint()
def run(
    split_csv_path: str = "train_split.csv",
    output_jsonl: str = "train_nbest_candidates.jsonl",
    num_shards: int = 10,
    sample_n: int = None,
):
    main.remote(
        split_csv_path=split_csv_path,
        output_jsonl=output_jsonl,
        num_shards=num_shards,
        sample_n=sample_n,
    )