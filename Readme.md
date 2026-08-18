# N-Best Rescoring for Low-Resource Bangla ASR

## What this project is testing

Whisper (and ASR models generally) perform worse on low-resource languages
like Bangla than on English. Fully fine-tuning a model to close that gap is
expensive and has to be repeated whenever the base checkpoint or data
changes. This project tests a cheaper alternative: **N-best rescoring** —
instead of trusting Whisper's single top transcription, ask it for its top 5
candidate guesses, then use a second, lighter-weight method to pick the best
one out of those 5.

The question the paper is actually answering: *is there a smarter way to
pick, or does Whisper's own top pick already win most of the time?*

## Pipeline overview

```
raw 1-hour audio (train_NNN.wav)
+ untimed paragraph transcript (train_NNN.txt)
            │
            ▼
  preprocess_chunk_align.py   →  aligned_chunks.csv
  (VAD chunking → Whisper transcribe → fuzzy-align to ground truth)
            │
            ▼
  data_loader.py              →  train_split.csv / eval_split.csv
  (group-aware split by source recording, prevents leakage)
            │
            ▼
  inference.py                →  nbest_candidates.jsonl
  (batched beam search, 5 candidates per utterance)
            │
            ▼
  evaluation.py                →  baseline_oracle_results.json
  (1-best WER/CER vs. Oracle WER/CER — the actual research question)
```

## File by file

### `preprocess_chunk_align.py`
Builds a labeled dataset from raw material that has no timestamps at all.

- **`VadChunker`** — runs Silero VAD to find speech, greedily packs it into
  chunks up to 25s, cutting only at silence so words don't get split across
  chunk boundaries.
- **`ChunkingPreprocessor`** — transcribes each chunk with a Bangla-adapted
  Whisper-medium checkpoint in FP16. `transcribe_batch` batches multiple
  chunks per `generate()` call for GPU throughput, and correctly passes
  `attention_mask` so padded (shorter) chunks in a batch don't get attended
  to as if they were real audio.
- **`align_chunks_sequential`** — since the ground-truth transcript is one
  untimed paragraph, this fuzzy-matches each chunk's ASR hypothesis against
  a sliding window of that paragraph (RapidFuzz), advancing a sequential
  pointer through the corpus. A confidence gate (`MIN_ALIGNMENT_SCORE = 50`)
  stops a single bad hypothesis from derailing every chunk after it — low-
  confidence matches don't advance the pointer.
- **Resumable**: checkpoints completed source files individually; a crash
  mid-file doesn't leave a half-written, mislabeled row, and reruns skip
  already-completed files.

Output: `aligned_chunks.csv` (`audio_path, gt, align_score`) plus a full
audit manifest and a stats file.

### `data_loader.py`
Loads `aligned_chunks.csv` and produces train/eval splits.

- Recovers each chunk's source recording from its filename (`train_001_000`
  → `train_001`).
- Splits with `GroupShuffleSplit` **at the source-recording level**, not the
  chunk level — this is the important part. Chunks from the same hour-long
  recording share a speaker, recording conditions, and often vocabulary; a
  naive row-level split would let eval-time performance benefit from context
  the model effectively already saw in training.
- Asserts zero source-file overlap between train and eval as a hard check,
  not just a hope.

### `inference.py`
The actual experiment: generates N-best candidates per utterance.

- **`NBestGenerator`** — loads the Whisper checkpoint, sets beam-search
  parameters directly on `model.generation_config` (required for beam search
  to reliably engage — passing them only as `generate()` kwargs was found
  not to be enough), and forces Bangla transcription via
  `get_decoder_prompt_ids` rather than the `language=`/`lang_to_id` path,
  which some fine-tuned checkpoints' `generation_config.json` doesn't
  support.
- **`--diverse_beam_search`** — switches from plain (deterministic) beam
  search to stochastic beam search (`do_sample=True`, `num_beams>1`).
  Group Beam Search was tried first but turned out to require a Hub-hosted
  custom generation strategy that wasn't usable in this environment, so
  sampling is the fallback diversity mechanism.
- Batched, resumable at the utterance level (checkpoint + immediate flush
  per batch), with a `--limit` timing-test mode to extrapolate full-run cost
  before committing GPU time.

Output: `nbest_candidates.jsonl` — 5 ranked, scored candidates per utterance.

### `evaluation.py`
Scores the N-best output.

- **1-best baseline**: WER/CER of Whisper's own top-ranked candidate.
- **Oracle**: WER/CER if you always picked whichever of the 5 candidates
  happened to be closest to the ground truth — the theoretical ceiling any
  rescoring method could reach.
- Shared normalization applied to both hypothesis and reference before
  scoring, so comparisons aren't confounded by punctuation/whitespace
  differences.
- Warns automatically if the Oracle/1-best gap is under 2 WER points, since
  that means the 5 candidates are too similar for any rescoring method to
  meaningfully choose between.

## Current status: the diversity collapse finding

On two pilot subsets (58 and 100 utterances), **Oracle WER equaled 1-best
WER to two decimal places** under plain beam search. Manually inspecting
candidates confirmed this wasn't a scoring bug — the 5 beams were near-
identical, with log-probabilities differing only in the 6th/7th decimal
place. Switching to stochastic sampling (`temperature=1.0`) didn't break it.

This matters because Variants 1–3 of the planned rescoring comparison
(LM-likelihood rescoring, MBR decoding, and a hybrid of the two) can only
*select* among existing candidates. If the candidates are functionally
identical, there's nothing for any of those methods to find — a small or
zero WER improvement from them wouldn't reflect on the methods' quality, it
would just reflect an already-collapsed candidate pool.

## What's actually blocking progress right now

Not infrastructure — the core open question is whether this collapse is
avoidable at all:

1. **`temperature=1.8` pilot** — not yet run. Cheapest next test.
2. **Nucleus (top-p) sampling pilot** — not yet run. Structurally different
   from raising temperature on a peaked softmax, worth trying independently.
3. **Full-corpus N-best generation + Variants 1–3** — intentionally on hold
   until 1–2 resolve, since running them now would likely just confirm what
   the pilots already suggest, at real GPU cost.
4. **Variant 4 (learned reranker)** and **GEC (generative correction)** —
   worth piloting regardless of how 1–2 turn out. GEC in particular doesn't
   depend on the beam pool's diversity at all, since it synthesizes a
   correction rather than selecting among existing candidates.

If the collapse holds under all of 1–2, the paper's contribution shifts from
"we improved Bangla ASR via rescoring" to a diagnostic finding: standard
N-best rescoring has near-zero headroom on confidently-decoded, clearly
narrated Bangla speech, independent of how sophisticated the rescoring
method is — which is a real, defensible result in its own right, and points
toward GEC as the more promising direction for future work.

## Known issues already fixed
- `attention_mask` wasn't being passed during batched Whisper transcription
  in an earlier draft of `preprocess_chunk_align.py`, which risked
  batch-composition-dependent transcription quality on shorter chunks.
  Fixed.
- `inference.py`'s beam-search parameters weren't reliably taking effect
  when passed only as `generate()` kwargs. Fixed by setting them on
  `generation_config` directly.

## Explored and set aside
- **CTC forced alignment** as a replacement for the VAD + Whisper +
  fuzzy-match preprocessing stage — would remove the alignment step's
  circularity (alignment quality currently depends on the same ASR system
  being evaluated), but judged to need a near-complete rebuild of the
  preprocessing stage for the payoff it offers right now. Left as a
  documented future-work item rather than pursued further.