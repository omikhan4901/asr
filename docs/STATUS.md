# CSE499 Bangla ASR — research codebase status and handoff

This file is the standalone handoff for the dual-model / SWER / GRPO phase of the project. It assumes you have **not** been in the implementation conversation. Read it before trusting `compare_models.py` SWER columns or running any GRPO job.

**Language of the project:** Bangla is the baseline and primary track. English clinical corpora (MedDialog-EN-ASR, PriMock) are an optional parallel track that is **not implemented**. Do not pivot the defaults.

**Published Whisper baseline (must still reproduce):**

| Metric | Eval split, N=2,271 |
|---|---|
| 1-best WER | 34.85% (CER 14.12%) |
| MBR WER | 32.29% (CER 13.08%) |
| Oracle WER | 28.72% (CER 11.24%) |
| MBR headroom recovery | ~41.8% |

Those numbers come from the existing Whisper JSONL + `MBRSelector` path. New code was added *around* that path. `src/bangla_asr/models/autoregressive/whisper_model.py` and `src/bangla_asr/rescoring/mbr.py` were not rewritten.

---

## 1. Current state of every module

Legend: **trustworthy** = covered by unit tests and safe to use for the published baseline. **usable / unvalidated** = implemented and tested on synthetics, not yet checked on your real transcripts. **stub / placeholder** = API only, or seeded data that must be replaced.

### Acoustic models

| Module | State | Notes |
|---|---|---|
| `models/base.py` (`ASRModel`, `Hypothesis`) | trustworthy | Shared interface. Wire format is still `{"text","logprob","rank"}`. |
| `models/autoregressive/whisper_model.py` | trustworthy | Original loader. Untouched. |
| `models/autoregressive/generator.py` | trustworthy | Optional `num_candidates`; omitting it is the published 12-beam path. |
| `models/autoregressive/whisper_adapter.py` | trustworthy | Thin `ASRModel` wrapper. Tests use a fake generator (no weight download). |
| `models/ctc/wav2vec2_model.py` | usable / unvalidated | Greedy when `n==1`. `n>1` requires `pyctcdecode` and **raises** if it is missing. Not run on the 2,271-utt set yet. |
| `models/ctc/beam_decoder.py` | usable / unvalidated | Optional KenLM. No Bangla LM is bundled. |
| `models/registry.py` | trustworthy | `build_asr_model("whisper"\|"wav2vec2_ctc")`. |
| `models/bert_reranker/` | trustworthy (existing) | Unchanged. Still a third *rescoring* column, not a third acoustic model. |

CTC scores are **not** on the Whisper log-prob scale. Do not reuse MBR temperature 0.5 on Wav2Vec2 and call it the same method. `pipeline/09_compare_acoustic_models.py` exposes a separate `--wav2vec2_mbr_temperature`.

Whisper 1-best in comparison tables is **rank 0** of the N-best list (published definition). CTC 1-best is stored `greedy_text` when present, because KenLM beam-1 can differ from greedy.

### Data / preprocess

| Module | State | Notes |
|---|---|---|
| `preprocessing/vad.py`, `aligner.py` | trustworthy | Existing Bangla long-form chunking. |
| `dataset/manifest.py`, `splitter.py` | trustworthy | Existing CSV / JSONL contracts. |
| `dataset/loaders.py` | mixed | `bangla_manifest` wraps the aligned CSV. `meddialog_en_asr` / `primock` raise `NotImplementedError` on purpose. |

### Rescoring and eval

| Module | State | Notes |
|---|---|---|
| `rescoring/mbr.py` | trustworthy | Probability-weighted MBR. Temperature default 0.5. |
| `rescoring/oracle.py` | trustworthy | 1-best (rank 0) and oracle. Do not change this if you want 34.85 / 28.72. |
| `eval/compare_models.py` | usable / unvalidated | Side-by-side WER + SWER. SWER columns are only as good as the metric (see below). |
| `pipeline/03_generate_nbest.py` … `08_evaluate_benchmarks.py` | trustworthy | Published runners. |
| `pipeline/03b_generate_wav2vec2_nbest.py`, `04b_modal_wav2vec2_nbest.py`, `09_compare_acoustic_models.py` | usable / unvalidated | New. Need real CTC JSONL. |

### SWER (core research metric)

| Module | State | Notes |
|---|---|---|
| `common/metrics.py` (`align_tokens`) | trustworthy | Same Wagner–Fischer recurrence as WER; backtrace added for SWER. |
| `metrics/swer.py` | usable / unvalidated | Formula implemented; **not** signed off against your hand labels. |
| `metrics/garbled.py` | usable / unvalidated | Default detector is `composite`. Swappable. |
| `metrics/lexicons/placeholder_bangla.yaml` | **placeholder** | A handful of example clinical terms. Not a gold lexicon. `না` is marked critical and will over-fire. |
| `metrics/lexicons/starter_bangla_wordlist.txt` | **starting point** | Small frequency-style list plus test/clinical tokens. Not a dictionary. |
| `tests/fixtures/swer_sanity_examples.json` | **empty scaffold** | `ready: false`. Fill this before trusting SWER. |
| `tests/test_swer_sanity.py` | skipped until you fill the fixture | Rank-order check, not exact-value matching. |

SWER is **not** Roy 2021 Semantic WER, not HALAS, not “WER is Unaware,” and not Answer Error Rate. It is a clinical-severity-weighted edit rate with an explicit multiplier table.

### Correction / RL

| Module | State | Notes |
|---|---|---|
| `training/reward.py` | usable / unvalidated | `SWER(asr, gt) - SWER(correction, gt)`. Same `compute_swer` as eval. |
| `training/train_grpo.py` | implemented, **not run** | Full GRPO loop. Do not launch until SWER sanity passes. |
| `training/callbacks.py` | trustworthy | JSONL + CSV logs only. No W&B. |
| `training/modal_train.py` | implemented, **not run** | Same Modal volume pattern as N-best. |
| `pipeline/10_train_grpo.py` | implemented, **not run** | Local entrypoint. |
| `correction/reasoning_model.py` | thin wrapper | Loads a causal LM. Untested with real VibeThinker weights. |
| `correction/react_scaffold.py` | **stub** | Thought/Action/Observation parser. Tools do not see audio, N-best, or a clinical KB. |

Default policy id is the string `WeiboAI/VibeThinker-3B`. **Confirm this Hugging Face id yourself.** It may be wrong or may be 1.5B rather than 3B. The field is config-swappable.

### Web demo / figures

Unchanged. Not part of the SWER/GRPO research path.

---

## 2. Every judgment call (one list)

These are decisions, not “just how the code fell out.” Several need supervisor review.

### 2.1 SWER edit multipliers (your draft, implemented literally)

| Event | Weight |
|---|---|
| Critical clinical substitution | 5.0 |
| Garbled / unintelligible span | 5.0 |
| Minor lexical substitution | 1.0 |
| Critical deletion | 4.0 |
| Non-critical deletion | 1.0 |
| Hallucinated insertion | 3.0 |
| Filler / repeated insertion | 1.0 |

Token match is **exact after** `normalize_bangla_text` (nukta + punctuation strip). No fuzzy token matching. That is a later research question.

### 2.2 Denominator (interpreted — the draft did not specify token mass)

SWER = (sum of edit weights) / (sum of **reference-token** weights).

- Critical reference token mass = `critical_reference_weight` = **5.0**
- Other reference tokens = 1.0
- Insertions add to the **numerator only**

This was chosen so a one-token critical substitution is 5/5 = 1.0 (WER-like range), not 5/1 = 5.0. If you change the mass, all SWER numbers move. Empty references have denominator 0 → SWER is **NaN** (same as WER).

### 2.3 Critical deletion 4.0 vs substitution 5.0 (unresolved clinical decision)

For one critical reference token:

- substitution SWER = 5/5 = **1.0**
- deletion SWER = 4/5 = **0.8**

Implication: **dropping a critical term is scored as slightly less severe than getting it wrong.** That pairing was an artifact of the first draft numbers, not a decision you made with your supervisor. The weights were **not** equalized. The asymmetry is documented on `SWERConfig` in `src/bangla_asr/config.py`. Decide with your supervisor whether to keep, equalize, or reverse it.

### 2.4 Garbled detection (default: `composite`)

Earlier, garbled only fired on a tiny `garbled_tokens` list. That hid CTC collapse. That is fixed.

**Options implemented** (`SWERConfig.garbled_detector`):

| Name | Rule | FP risk | FN risk | Needs wordlist |
|---|---|---|---|---|
| `explicit` | lexicon list only | very low | near-total on real CTC | no |
| `character` | illegal Bangla orthography (3+ char runs, leading matra/virama, stacked matras/viramas, trailing virama) | low | misses “looks like a word but isn’t” | no |
| `wordlist_oov` | any token not in the wordlist | **high** until the list is large | low | yes |
| `edit_distance` | OOV **and** char-Levenshtein to nearest listed word > threshold (default 2) | medium | medium | yes |
| `composite` (**default**) | explicit **or** illegal orthography **or** (OOV and far from the wordlist) | medium | medium | yes (starter list ships) |

**Why composite is the default:** CTC collapse is often illegal orthography *or* far from any real word. Pure OOV would mark unseen names/inflections/drugs as garbled (false positives). Pure explicit list would miss almost everything. You can swap the strategy without touching the rest of SWER.

The shipped wordlist (`starter_bangla_wordlist.txt`) is a **starting point**. Replace it with a real frequency list before a paper claim.

`treat_oov_as_garbled=False` remains the default. Turning it on with the starter list will over-flag ordinary Bangla.

### 2.5 Garbled cost is per *span*, not per token

Adjacent garbled hypothesis tokens (consecutive `hyp_index`) are one span. The span is charged `garbled_span_weight` (5.0) **once**. Continuations are kept in the alignment with weight 0. A 1-token collapse and a 3-token adjacent collapse of the same reference therefore contribute the same garbled mass. Test: `tests/test_garbled.py::test_one_token_and_three_token_garbled_span_cost_the_same`.

Garbled insertions (collapsed junk with no source token) join that span; they are not also charged as 3.0 hallucinations.

Non-adjacent garbled tokens are separate spans (5.0 each).

### 2.6 Alignment tie-break

When several Levenshtein operations have the same cost, backtrace prefers **diagonal (equal/substitute), then delete, then insert**. WER is unchanged (same distance). SWER can change because different tokens get different weights. This is standard but arbitrary; if you change it, SWER numbers can move without WER moving.

“Repeated insertion” is a consecutive duplicate of a **neighbor in the hypothesis**, because the aligner often inserts the first copy, not the second.

### 2.7 Reward shape

`reward = SWER(asr_hypothesis, gt) - SWER(correction, gt)`, then `reward_scale` and `reward_clip` (config, not hidden constants). This rewards *improvement over the ASR baseline*, so easy already-correct utterances do not dominate.

### 2.8 Empty-reference handling in GRPO

SWER(NaN) is not a number you can train on.

- **Default now:** `GRPOConfig.exclude_empty_reference_from_batch = True`. Those pairs are **dropped** from the batch. Logs include `excluded_empty_reference` and `n_rewarded`.
- **Old behaviour (opt-in):** set the flag False to map them to reward 0.0, which **dilutes** the batch mean.

`swer_improvement_reward(...)` returns `None` when a pair is excluded.

### 2.9 GRPO vs MGPO

GRPO was chosen: one group of G on-policy rollouts, group-relative advantages, no critic. Fits a ~3B policy on one Modal GPU. MGPO (multi-objective / multi-group) is a later axis, after SWER is frozen.

### 2.10 Logging

JSONL + CSV only. No Weights & Biases, not even as an optional flag.

### 2.11 English clinical data

Loaders exist as named hooks and raise `NotImplementedError`. They must not change Bangla defaults. An English SWER lexicon does not exist and should not be faked from general knowledge.

### 2.12 Config system

Dataclass defaults are the reproducibility contract for 34.85 → 32.29. YAML files under `src/bangla_asr/configs/` are overlays, not Hydra. Do not change `WhisperConfig` / `MBRConfig` defaults to “clean things up.”

---

## 3. What to do next (in this order)

1. **Hand-label SWER sanity examples.** Open `tests/fixtures/swer_sanity_examples.json`. Fill ~10 real `(asr_hypothesis, ground_truth, manual_severity)` triples from Whisper **and** Wav2Vec2 (include at least one CTC garble and one Whisper fluent error). Set `"ready": true`. Run `python tests/test_swer_sanity.py`. If rank order fails, the metric is wrong for your domain — fix the lexicon/weights/detector **before** any training.
2. **Supervisor sign-off on deletion vs substitution.** Show them §2.3 (0.8 vs 1.0). Change `critical_deletion_weight` / `critical_substitution_weight` in `SWERConfig` only after that conversation.
3. **Replace the starter Bangla wordlist.** Point `SWERConfig.garbled_wordlist_path` at a real unigram list (IndicCorp / OSCAR / your train-set vocabulary). Re-run garbled tests and the sanity harness.
4. **Replace the PLACEHOLDER clinical lexicon.** `placeholder_bangla.yaml` is not publishable. Build the real table from domain reference material (drug names, negation, dose patterns, opposite findings). Keep English clinical terms in a **separate** file if you ever add that track.
5. **Generate Wav2Vec2 N-best on the held-out split** (`pipeline/03b` locally or `04b` on Modal). Then `pipeline/09_compare_acoustic_models.py`. Read CTC vs Whisper SWER as “error type,” not just “who wins.”
6. **Confirm the policy checkpoint id.** Check Hugging Face for VibeThinker (3B vs 1.5B). Put the real id in `GRPOConfig.model_id`.
7. **Only then:** a **tiny** Modal GRPO smoke test (`max_steps` small, tiny JSONL). Watch `excluded_empty_reference`, reward monotonicity, and eval SWER. Do not burn a full run on an unchecked metric.
8. **Decide whether English clinical data is ever needed.** If yes, write a dedicated loader + English lexicon; do not reuse Bangla defaults.

Do **not** run GRPO/Modal training until step 1 passes on your labels.

---

## 4. External resources you still need to source

| Resource | Why | Status in repo |
|---|---|---|
| Real clinical severity lexicon (Bangla) | Core paper contribution | PLACEHOLDER only |
| Proper Bangla wordlist / frequency list | Garbled detection recall/precision | starter list only |
| Hand-labeled SWER triples | Sanity before training | empty fixture |
| Confirmation of `WeiboAI/VibeThinker-3B` | GRPO policy | string default only |
| Optional KenLM Bangla LM | CTC N-best with LM (changes the scientific claim) | not bundled |
| Wav2Vec2 eval JSONL on the 2,271-utt set | Dual-model table | not generated here |
| Optional: English MedDialog / PriMock + English lexicon | Parallel track | hooks only |
| `pyctcdecode` (and PyYAML if you edit YAML lexicons) | CTC N-best / YAML load | extras: `pip install -e ".[ctc]"` and `".[config]"` |

---

## 5. How to run everything from scratch

Python 3.9+ recommended. From the repo root:

```bash
pip install -e .
pip install -e ".[audio,vad,ctc,config,distributed,dev]"
```

On Windows, set `PYTHONPATH=src` if you run scripts without the editable install.

### 5.1 Reproduce the published baseline (regression check on a new machine)

You need the **existing** eval N-best JSONL (or regenerate it with the same Whisper defaults: 12 beams, sampling temperature 0.7):

```bash
python pipeline/05_rescore_mbr.py ^
  --nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl ^
  --output_dir data/processed/results/mbr ^
  --temperature 0.5

python pipeline/08_evaluate_benchmarks.py ^
  --eval_nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl ^
  --output_dir data/processed/results/benchmark
```

Expect 1-best ≈ 34.85% and MBR ≈ 32.29% on the full 2,271-utt eval set. If you only have a subset, the **absolute** WER will differ; the MBR/oracle machinery should still match `tests/test_mbr.py` and `tests/test_smoke.py`.

To regenerate Whisper N-best (GPU; expensive):

```bash
python pipeline/03_generate_nbest.py --split_csv data/processed/splits/eval_split.csv --num_beams 12 --temperature 0.7
```

or Modal:

```bash
modal run pipeline/04_modal_nbest.py --split-csv-path eval_split.csv --output-jsonl eval_nbest_candidates.jsonl --num-shards 10
```

Do **not** change `WhisperConfig` defaults if you want that regression.

### 5.2 Full Bangla pipeline (existing numbered scripts)

1. `pipeline/01_preprocess.py` — VAD chunk + align
2. `pipeline/02_split_dataset.py` — group-aware split
3. `pipeline/03_generate_nbest.py` — Whisper N-best
4. `pipeline/04_modal_nbest.py` — same, Modal
5. `pipeline/05_rescore_mbr.py` — MBR
6. `pipeline/06_train_bert_reranker.py` / `07_rescore_bert.py` — optional BERT column
7. `pipeline/08_evaluate_benchmarks.py` — 1-best / MBR / BERT / oracle

### 5.3 Dual-model + SWER (new)

```bash
pip install pyctcdecode
python pipeline/03b_generate_wav2vec2_nbest.py --split_csv data/processed/splits/eval_split.csv --n 12
python pipeline/09_compare_acoustic_models.py --whisper_jsonl data/processed/nbest/eval_nbest_candidates.jsonl --wav2vec2_jsonl data/processed/nbest/eval_wav2vec2_nbest.jsonl
```

Configs:

- Whisper defaults: `src/bangla_asr/config.py` (`WhisperConfig`, `MBRConfig`)
- CTC: `Wav2Vec2Config` / `src/bangla_asr/configs/model/wav2vec2.yaml`
- SWER: `SWERConfig` / `src/bangla_asr/configs/metrics/swer.yaml`
- GRPO: `GRPOConfig` / `src/bangla_asr/configs/training/grpo.yaml`

### 5.4 GRPO (only after sanity)

```bash
python pipeline/10_train_grpo.py --train_jsonl data/processed/correction/train.jsonl --eval_jsonl data/processed/correction/eval.jsonl --max_steps 5
```

Modal (same `bangla-asr-data` volume as N-best):

```bash
modal run src/bangla_asr/training/modal_train.py --train-jsonl /data/train_correction.jsonl --max-steps 5
```

Logs: `logs/grpo/train_log.jsonl` and `.csv`. Watch `excluded_empty_reference`.

### 5.5 Tests

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

`test_swer_sanity.py` should **skip** until you set `ready: true` and fill the fixture. A skip here is not a failure; it is a reminder.

---

## 6. File map (new vs old)

New or substantially extended for this phase:

```
src/bangla_asr/models/base.py
src/bangla_asr/models/registry.py
src/bangla_asr/models/autoregressive/whisper_adapter.py
src/bangla_asr/models/ctc/
src/bangla_asr/metrics/swer.py
src/bangla_asr/metrics/garbled.py
src/bangla_asr/metrics/severity_lexicon.py
src/bangla_asr/metrics/lexicons/
src/bangla_asr/eval/
src/bangla_asr/training/
src/bangla_asr/correction/
src/bangla_asr/dataset/loaders.py
src/bangla_asr/configs/
pipeline/03b_generate_wav2vec2_nbest.py
pipeline/04b_modal_wav2vec2_nbest.py
pipeline/09_compare_acoustic_models.py
pipeline/10_train_grpo.py
tests/test_*.py (except the original test_smoke.py, which still exists)
tests/fixtures/swer_sanity_examples.json
docs/STATUS.md
```

Intentionally **not** rewritten: `whisper_model.py`, `mbr.py`, `oracle.py` 1-best definition, web demo, figure scripts, Whisper/MBR default numbers.

---

## 7. How to explain this to a supervisor in one minute

The published result is still inference-only MBR on Whisper N-best (34.85 → 32.29, ~42% of oracle headroom, no weight updates). The next research bet is (1) a second acoustic model (Wav2Vec2 CTC) under the same rescoring interface, and (2) a **clinical-severity-weighted** error rate (SWER) that is not Semantic WER, plus an optional small-model correction policy trained with GRPO on SWER *improvement*. SWER is implemented and unit-tested on synthetics, but the clinical lexicon, Bangla wordlist, deletion/substitution weights, and hand-labeled sanity set are not signed off. Training has not been run.
