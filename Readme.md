# Inference-Only Rescoring for Bangla Long-Form ASR

Companion code for *"Inference-Only Rescoring for Bangla Long-Form ASR:
Characterizing Oracle Headroom and MBR Consensus Recovery"* (CSE499
capstone, North South University).

**Headline result (eval split, N=2,271):** 1-best WER 34.85% → MBR-rescored
32.29% → Oracle ceiling 28.72%, with zero gradient updates. MBR recovers
~41.8% of available oracle headroom and lands within ~1 point of a cited
LoRA fine-tune baseline (31.32%, Bhuiyan et al., arXiv:2605.08214).

A live demo (upload a video, get back Bangla `.srt` subtitles built on this
exact pipeline) is deployed at: **[add your Vercel URL here once live]**

---

## What this is

Standard practice for improving low-resource ASR is fine-tuning — full
weight updates or LoRA — which costs real GPU-hours and risks catastrophic
forgetting. This project asks a narrower question instead: **without
touching a single model weight**, how much error can be recovered just by
generating multiple candidate transcriptions per utterance and picking a
better one via consensus?

The short answer: about 42% of a real, measured headroom — but not more,
and the paper is explicit about why (Section V, "Discussion and
Limitations").

## Pipeline

```
121 raw Bangla recordings + untimed reference transcripts
        │
        ▼
PHASE 1 — preprocess_chunk_align.py
  Silero VAD chunking (0.5–25.0s, cut only at silence)
  → rough ASR pass (alignment hypothesis only)
  → sequential fuzzy alignment (RapidFuzz, τ ≥ 50 confidence gate)
        │
        ▼
aligned_chunks.csv — 11,129 utterances (72.7% retention of 15,312 raw chunks)
        │
        ▼
data_loader.py — group-aware split (by SOURCE RECORDING, not chunk,
                  to prevent train/eval leakage)
        │
   ┌────┴────┐
   ▼         ▼
train_split.csv    eval_split.csv
(8,858 utt.)        (2,271 utt.)
   │                    │
   ▼                    ▼
PHASE 2 — modal_inference.py (both splits, identical config)
  10 shards → 10x A10G containers, parallel
  num_beams=12, sampling_temperature=0.7, batch_size=2
  per-utterance write + volume.commit() (resumable — a container
  timeout only loses the in-flight batch, not completed work)
   │                    │
   ▼                    ▼
train_nbest_candidates.jsonl   eval_nbest_candidates.jsonl
   │                    │
   └─────────┬──────────┘
             ▼
PHASE 3 — evaluation.py (1-best baseline + Oracle ceiling)
          mbr_rescore.py (probability-weighted MBR consensus)
             │
             ▼
      Headline results (see paper Table I / Fig. 6)
```

Config (`num_beams=12`, `temperature=0.7`) was not guessed — it's the
winner of a 100-utterance calibration sweep across 4 configs (`figures2/`
Fig. 4 in the paper), chosen for maximizing Oracle headroom without
degrading 1-best WER.

## Repo layout

```
src/
  preprocess_chunk_align.py   Phase 1 — chunking, transcription, fuzzy alignment
  data_loader.py               Group-aware train/eval split
  inference.py                 Single-machine N-best generation (local/Kaggle reference impl.)
  modal_inference.py           Distributed N-best generation on Modal (actual production path)
  evaluation.py                WER/CER, 1-best baseline, Oracle ceiling
  mbr_rescore.py                Probability-weighted MBR consensus (Eq. 1 in the paper)
  rescoring.py                  Scaffolding for LM/hybrid/learned-reranker variants
                                 (NOT used in the final paper — see Future Work)
figures/
  fig_empirical_pilot_wer.py    Fig. 6 — WER comparison bar chart
  dataset_funnel.py             Fig. 3 / Fig. 4 — dataset yield + calibration sweep
  fig_preprocessing_pipeline.drawio
  fig_nbest_generation_pipeline.drawio
  fig_system_overview.drawio    Fig. 1 / Fig. 2 in the paper
web/
  backend/
    app.py                      Deployable Modal app (FastAPI + GPU worker)
    local_server.py             Local-only test server, same API contract as app.py
    pipeline.py                 SubtitleGenerator — reuses the exact confirmed
                                 config + MBR formula from src/, per-video not sharded
    srt_utils.py                Chunk list -> valid .srt file
  frontend/
    index.html, script.js, style.css   Static upload UI, polls job status
data/                           NOT committed — see .gitignore below
```

## Running it

**Phase 1 (preprocessing):**
```
python src/preprocess_chunk_align.py --audio_dir <path> --annotation_dir <path> --output_dir data/processed/chunks
```

**Split:**
```
python src/data_loader.py --aligned_csv data/processed/chunks/aligned_chunks.csv --output_dir data/processed/splits
```

**Phase 2 (N-best generation, Modal):**
```
modal run modal_inference.py --split-csv-path train_split.csv --output-jsonl train_nbest_candidates.jsonl
modal run modal_inference.py --split-csv-path eval_split.csv --output-jsonl eval_nbest_candidates.jsonl
```

**Phase 3 (scoring):**
```
python src/evaluation.py --nbest_jsonl eval_nbest_candidates.jsonl --output_dir data/processed/results
python src/mbr_rescore.py --nbest_jsonl eval_nbest_candidates.jsonl --output_dir data/processed/results/mbr
```

**Demo backend (local test):**
```
cd web/backend
uvicorn local_server:app --reload --port 8000
```

**Demo backend (Modal deploy):**
```
modal deploy web/backend/app.py
```
Copy the printed URL into `web/frontend/index.html`:
```html
<script>window.ASR_API_BASE = "https://<your-modal-url>.modal.run";</script>
```

**Demo frontend (Vercel):** deploy `web/frontend/` as a static site. No
build step required — it's plain HTML/CSS/JS.

## Known limitations (from the paper, Section V–VI — stated, not hidden)

- MBR recovers ~42% of Oracle headroom, not more — it operates purely on
  acoustic probability and text similarity, so it can't distinguish a
  genuinely correct consensus from several beams confidently agreeing on
  the same hallucination.
- The remaining ~58% headroom is explicitly scoped as future work:
  KenLM n-gram interpolation and a learned (GBDT/LLM-based) reranker over
  acoustic confidence + entropy features. `src/rescoring.py` has scaffolding
  for both, deliberately not run for the paper's final numbers due to time
  constraints.
- Full end-to-end fine-tuning is explicitly out of scope for this paper by
  design (that's the whole point — see Abstract) but is the natural next
  step, and is already known to reach 24.41% WER (cited, not reproduced
  here).
- The demo's subtitle segments are chunk-level (up to 25s), not
  sentence-level — noted directly in the frontend's footer.

## Citation context

Baseline (34.07%), LoRA fine-tune (31.32%), and full fine-tune (24.41%)
comparison numbers are cited from Bhuiyan et al., "Bangla-WhisperDiar,"
arXiv:2605.08214 — not reproduced by this project. Marked with `*` in every
figure that includes them.