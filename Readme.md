# Bangla Long-Form ASR: Inference-Only Rescoring & Discriminative Reranking

**Group textbook (read this if you are new):** `cd handbook && npm install && npm run dev`, then open http://localhost:3000. **Engineering handoff:** [`docs/STATUS.md`](docs/STATUS.md). It lists what is trustworthy, every SWER/GRPO judgment call, remaining resources to source, and the ordered next-step list. This README is the command reference for the published Whisper baseline plus the newer dual-model / SWER / GRPO stages.

Companion codebase for *\"Inference-Only Rescoring for Bangla Long-Form ASR: Characterizing Oracle Headroom and MBR Consensus Recovery\"* (CSE499 capstone, North South University).

**Headline result (eval split, N=2,271):** 
- **1-Best Baseline WER:** 34.85% (CER: 14.12%)
- **MBR-Rescored WER:** 32.29% (CER: 13.08%)
- **Oracle Ceiling WER:** 28.72% (CER: 11.24%)
- **Headroom Recovery:** Minimum Bayes Risk (MBR) recovers **~41.8%** of the available oracle headroom without a single parameter update, approaching the cited LoRA fine-tune baseline (31.32%, Bhuiyan et al., arXiv:2605.08214).

---

## Architectural Layout

The repository is organized as a modular Python package (`bangla_asr`) with dedicated subpackages for autoregressive candidate generation (Whisper) and discriminative sequence reranking (BanglaBERT), alongside unified CLI scripts and a production web service:

```
bangla_asr/
├── pyproject.toml                     # Standard PEP 517/518 build specification
├── src/
│   └── bangla_asr/                    # Core Python Library
│       ├── config.py                  # Centralized dataclass configurations
│       ├── common/                    # Shared utilities
│       │   ├── audio.py               # Robust audio loading (torchaudio/soundfile/librosa), silence padding
│       │   ├── text.py                # Bangla text normalization & nukta canonicalization
│       │   └── metrics.py             # Guarded WER/CER computation with jiwer
│       ├── preprocessing/             # Phase 1: VAD & Alignment
│       │   ├── vad.py                 # Silero VAD segmenter with strict max-duration subdivision
│       │   └── aligner.py             # Sequential RapidFuzz aligner with lookahead recovery
│       ├── dataset/                   # Dataset loaders & splitting
│       │   ├── splitter.py            # Group-aware train/eval partitioner (arbitrary digit chunk support)
│       │   └── manifest.py            # Ingestion & serialization for manifests and JSONL pools
│       ├── models/
│       │   ├── autoregressive/        # Autoregressive Whisper Pipeline
│       │   │   ├── whisper_model.py   # Model loader with FP16, SDPA, prompt ID resolution
│       │   │   ├── generator.py       # Batched N-best generator with sampling-based diverse beam search
│       │   │   └── modal_worker.py    # Parallel worker container for distributed Modal sharding
│       │   └── bert_reranker/         # Discriminative BERT Reranker
│       │       ├── model.py           # BanglaBERT regressor with acoustic search features
│       │       ├── dataset.py         # PyTorch Dataset for N-best candidate pools with empirical WER targets
│       │       ├── trainer.py         # Training loop with AdamW, linear schedule, and checkpointing
│       │       └── ranker.py          # Candidate scorer and minimum-error selector
│       └── rescoring/                 # Phase 3: Rescoring & Evaluation
│           ├── mbr.py                 # Temperature-calibrated MBR consensus rescorer
│           ├── oracle.py              # Baseline (1-best) and theoretical Oracle ceiling evaluator
│           └── evaluate_all.py        # Benchmark comparator (1-Best, MBR, BERT, Oracle)
├── pipeline/                          # Top-Level Pipeline Runners
│   ├── 01_preprocess.py               # Chunking, rough transcription, and sequential fuzzy alignment
│   ├── 02_split_dataset.py            # Group-aware train/eval split (zero speaker/recording leakage)
│   ├── 03_generate_nbest.py           # Local batched N-best candidate generation
│   ├── 04_modal_nbest.py              # Distributed multi-GPU candidate generation on Modal
│   ├── 05_rescore_mbr.py              # Run probability-weighted MBR consensus
│   ├── 06_train_bert_reranker.py      # Train BanglaBERT discriminative reranker on train candidates
│   ├── 07_rescore_bert.py             # Score and evaluate eval candidates with BERT reranker
│   └── 08_evaluate_benchmarks.py      # Benchmark comparison across all models
├── figures/                           # Publication Figures and Plotting
│   ├── scripts/                       # Clean IEEE-styled matplotlib generators
│   └── *.pdf                          # 300 DPI vector PDFs ready for paper embedding
└── web/                               # Full-Stack Subtitle Generation Web App
    ├── backend/
    │   ├── app.py                     # Serverless Modal backend (FastAPI + A10G worker)
    │   ├── local_server.py            # Thread-safe local development server with GPU mutex
    │   ├── pipeline.py                # SubtitleGenerator orchestrating VAD, Whisper, and MBR
    │   └── srt_utils.py               # SubRip (.srt) formatter with proper millisecond carries
    └── frontend/
        ├── index.html                 # Drag-and-drop web UI
        ├── script.js                  # Dynamic API detection, stage tracker, and polling loop
        └── style.css                  # UI styling
```

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/omikhan4901/asr.git
cd asr

# 2. Install package in editable mode
pip install -e .

# 3. Optional dependencies for web demo or plotting:
pip install fastapi uvicorn python-multipart modal matplotlib
```

---

## Running the Pipeline

### Phase 1: Audio Chunking and Sequential Alignment
```bash
python pipeline/01_preprocess.py \
  --audio_dir data/raw/train/audio \
  --annotation_dir data/raw/train/annotation \
  --output_dir data/processed/chunks
```

### Phase 2: Group-Aware Train/Eval Split
```bash
python pipeline/02_split_dataset.py \
  --aligned_csv data/processed/chunks/aligned_chunks.csv \
  --output_dir data/processed/splits
```

### Phase 3: N-Best Candidate Generation
**Option A: Local / Single-GPU execution:**
```bash
python pipeline/03_generate_nbest.py \
  --split_csv data/processed/splits/eval_split.csv \
  --output_jsonl data/processed/nbest/eval_nbest_candidates.jsonl \
  --num_beams 12 \
  --temperature 0.7
```

**Option B: Distributed execution on Modal (10x A10G parallel containers):**
```bash
modal run pipeline/04_modal_nbest.py \
  --split-csv-path train_split.csv \
  --output-jsonl train_nbest_candidates.jsonl \
  --num-shards 10
```

### Phase 4: MBR Consensus Rescoring
```bash
python pipeline/05_rescore_mbr.py \
  --nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl \
  --output_dir data/processed/results/mbr \
  --temperature 0.5
```

### Phase 5: Discriminative BERT Reranker
**1. Train the BERT Reranker on training candidate pools:**
```bash
python pipeline/06_train_bert_reranker.py \
  --train_nbest_jsonl data/processed/nbest/train_nbest_candidates.jsonl \
  --output_dir checkpoints/bert_reranker \
  --model_name csebuetnlp/banglabert \
  --epochs 5 \
  --batch_size 16
```

**2. Rescore and evaluate on the evaluation candidate set:**
```bash
python pipeline/07_rescore_bert.py \
  --eval_nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl \
  --checkpoint_dir checkpoints/bert_reranker \
  --output_dir data/processed/results/bert
```

### Phase 6: Comprehensive Benchmark Evaluation
Compare 1-Best Baseline, MBR Consensus, BERT Discriminative Reranking, and Oracle ceiling in a single command:
```bash
python pipeline/08_evaluate_benchmarks.py \
  --eval_nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl \
  --bert_checkpoint checkpoints/bert_reranker/best_model.pt \
  --output_dir data/processed/results/benchmark
```

---

## Dual acoustic models, SWER, and GRPO correction

Bangla remains the primary track. The published Whisper pipeline above is unchanged (`num_beams=12`, sampling temperature `0.7`, MBR temperature `0.5`). New modules live under `src/bangla_asr/` and do not replace `pipeline/03_generate_nbest.py` … `pipeline/08_evaluate_benchmarks.py`.

### Shared `ASRModel` interface
Whisper and Wav2Vec2 both implement `transcribe_nbest(audio, n) -> List[Hypothesis]`. Hypothesis dumps still use `{"text", "logprob", "rank"}` so `MBRSelector` is bit-compatible with the 32.29% path.

Install CTC extras before requesting Wav2Vec2 N-best (`n > 1`). Missing `pyctcdecode` raises; it does **not** repeat greedy 1-best.

```bash
pip install -e ".[ctc]"
python pipeline/03b_generate_wav2vec2_nbest.py \
  --split_csv data/processed/splits/eval_split.csv \
  --output_jsonl data/processed/nbest/eval_wav2vec2_nbest.jsonl \
  --model_id arijitx/wav2vec2-xls-r-300m-bengali
```

KenLM is optional (`--kenlm_model_path`). Acoustic-only beam search is still a real N-best list.

### Side-by-side evaluation
```bash
python pipeline/09_compare_acoustic_models.py \
  --whisper_jsonl data/processed/nbest/eval_nbest_candidates.jsonl \
  --wav2vec2_jsonl data/processed/nbest/eval_wav2vec2_nbest.jsonl \
  --output_dir data/processed/results/compare_models
```

CTC beam scores are **not** on the Whisper log-prob scale. Keep `--whisper_mbr_temperature 0.5` for baseline reproduction; set `--wav2vec2_mbr_temperature` separately. CTC 1-best uses stored `greedy_text` (KenLM beam-1 can differ from greedy).

### SWER (clinical-severity-weighted error rate)
This is **not** Roy 2021 Semantic WER. Implementation: `src/bangla_asr/metrics/swer.py` + `severity_lexicon.py`. The shipped table (`metrics/lexicons/placeholder_bangla.yaml`) is a **PLACEHOLDER**. English clinical lexicons are a parallel track and are not loaded by default.

### GRPO correction (SWER-improvement reward)
Reward is `SWER(asr, gt) - SWER(correction, gt)`, with `reward_scale` / `reward_clip` in config. Logs are JSONL + CSV only (no W&B).

```bash
python pipeline/10_train_grpo.py \
  --train_jsonl data/processed/correction/train.jsonl \
  --eval_jsonl data/processed/correction/eval.jsonl
```

Modal (same `bangla-asr-data` volume as N-best generation):

```bash
modal run src/bangla_asr/training/modal_train.py \
  --train-jsonl /data/train_correction.jsonl
```

`correction/react_scaffold.py` is a stub (text-only tools, no clinical KB). Use `--` / `prompt_style: react` only as an API experiment.

### Dataset loaders
`get_loader("bangla_manifest")` wraps the existing aligned CSV. `get_loader("meddialog_en_asr")` / `"primock"` raise `NotImplementedError` on purpose.

### Tests
```bash
set PYTHONPATH=src
python tests/test_smoke.py
python tests/test_asr_interface.py
python tests/test_wav2vec2_nbest.py
python tests/test_mbr.py
python tests/test_compare_models.py
python tests/test_swer.py
python tests/test_garbled.py
python tests/test_reward.py
python tests/test_loaders.py
python tests/test_react_scaffold.py
python tests/test_swer_sanity.py
```

---

## Live Web Application

### Local Testing
Start the thread-safe local server:
```bash
python web/backend/local_server.py
```
Open `web/frontend/index.html` in your web browser. The frontend will automatically detect the local API server on `http://127.0.0.1:8000`.

### Cloud Deployment (Modal)
Deploy the scalable serverless app to Modal:
```bash
modal deploy web/backend/app.py
```

---

## Generating Figures
All research figures are generated as publication-ready vector PDFs:
```bash
python figures/scripts/plot_funnel.py
python figures/scripts/plot_diversity_sweep.py
python figures/scripts/plot_wer_benchmarks.py
python figures/scripts/plot_chunk_distributions.py
```
Outputs are saved directly to `figures/`:
- `figures/wer_comparison.pdf`
- `figures/oracle_gap_recovery.pdf`
- `figures/beam_diversity_sweep.pdf`
- `figures/dataset_funnel.pdf`
- `figures/fig_alignment_scores.pdf`
- `figures/fig_chunk_durations.pdf`
