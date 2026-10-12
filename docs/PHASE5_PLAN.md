# Phase 5 plan: the released model, whole-recording evaluation, release

> **Goal (Omi, 2026-10-12): the lowest WER from one fast model; release questions later.**
> **Decision 2026-10-12 (Omi): one plan, no extras.** The final model is **M5-full** (called M7 below) = M5's recipe exactly (full FT, lr 1e-5, 2 epochs, no
> selection knobs) on ALL the data we have (156 h: DL Sprint 59.6 + CV 34.4 + Kathbath 33.9 + Shrutilipi 28.1).
> Evidence: M5 (10 h per source) is already within 0.4-0.6 of each specialist on CV / Kathbath / FLEURS; its only
> large gap is DL Sprint (13.84 vs 11.34), i.e. hours, not mixing. **Pre-registered:** DL Sprint eval <= 12.0, every
> other set within 0.5 of its specialist, average <= 12.5 (M5 13.18). **Kill rule:** average not significantly
> below M5 -> keep M5 and drop this route. Speed = M5's (one Whisper-medium; T4 RTF 0.035-0.038 batched). Everything below this note
> (new corpora, T1-T8, session D) is parked, not planned.
> **Second run, same recipe, only the data changes (Omi, 2026-10-12): M5-300** = M5-full's 156 h + OpenSLR 53
> (+ IndicVoices if it passes step 0), ~300 h, ~$11-14 est. Question: do new read-speech hours help beyond 156 h?
> **Kill rule:** M5-300's average not significantly below M5-full's -> keep M5-full, more hours of this kind do not help. Both runs ~$25-28 in
> November (friend's $30); Omi's ~$11 is the backup. Speed identical.
> **Two modes reported:** fast = M5-full greedy (default; live use); best = M5-full + ROVER on its own 12-best
> (offline batch jobs; expected −0.2 to −0.45 by analogy with M1; measured cost ~14x greedy on a T4: 552 vs 39 min
> for DL Sprint eval).

Phase 4 answered the method questions. Phase 5 builds the one model we release and finishes the paper.
Freeze: **Nov 12**; report: **Dec 1**. Times and costs marked *est.* are estimates.

## What Phase 4 decided for Phase 5

| Question | Answer (Phase 4) | Consequence for Phase 5 |
|---|---|---|
| Which training method? | full fine-tuning is best (11.34% eval; LoRA 11.76, decoder 12.99) | full fine-tuning |
| Which data? | each corpus wins its own test set; 10 h x 4 has the best average (13.18%) | train on **all four corpora together** (156 h, every hour they have) **plus new corpora up to ~250-300 h** |
| Harmonize the spelling? | no gain on DL Sprint, small loss on Common Voice (Gate 4) | **no** harmonization |
| Longer training / higher lr? | 4 epochs at 2e-5 over-fits (out-of-domain +1.4 to +2.3) | keep **1e-5, 2 epochs**; choose with a **mixed dev set** |
| Rescoring on top? | ROVER gives −0.45 after fine-tuning; word 4-gram LM reranking ties it (eval −0.29, FLEURS −0.47) | ROVER on the final model's 12-best (no extra LM work) |

## Steps

| # | Step | Who | Where | When | Time | Cost |
|---|---|---|---|---|---|---|
| 0 | Check the new corpora: real size, license, speakers vs our test sets (OpenSLR 53, IndicVoices Bengali; FLEURS train skipped, it would spoil FLEURS test) | Omi | web + PC | Oct | - | $0 |
| 0b | Prepare the new corpora (download, 16 kHz FLAC, same text rules, speaker filter) on a Modal CPU | Omi, **1 command** (+ accept the IndicVoices terms on HF if used) | Modal CPU | Oct | ~1-2 h est. | ~$0.5-1 est. |
| 1 | Merged training list (DL Sprint 59.6 h + CV 34.4 h + Kathbath 33.9 h + Shrutilipi 28.1 h + new corpora, **~250-300 h, each new source capped so DL Sprint stays >= ~20%**), mixed dev set (DL Sprint dev + CV dev + FLEURS dev + held-out Kathbath train speakers), `--source all` | Omi | PC (CPU) | Oct | - | $0 |
| 2 | Whole-recording evaluation code (VAD cuts the raw recording, the model transcribes, the text is joined, WER per recording on the 22 eval recordings) | Omi | PC | Oct | - | $0 |
| 3 | Kaggle job for baselines: IndicConformer + the released DL Sprint finalist models on the 4 test sets; whole-recording eval of zero-shot and M1 full | Omi, **1 Kaggle session** | Kaggle | Oct | ~6-9 h est. | $0 |
| 4 | Upload the merged list to the hub | **you run 1 command** | PC | Oct 31 | ~10 min | $0 |
| 5 | Train the released model (M7, ~250-300 h, 2 epochs at 1e-5) | **you run 1 command** | Modal (November credit) | Nov 1 | ~5-6 h est. | ~$11-14 est. |
| 6 | Send M7 to the hub (`to_hf --run m7_all`) | **you run 1 command** | Modal | Nov 1 | ~10 min | ~$0.05 |
| 7 | Score M7 + its 12-best for ROVER + whole-recording eval | **you start** (1-2 Kaggle sessions) | Kaggle | Nov 2-3 | ~10 h est. | $0 |
| 8 | Analysis, tables, figures, README, paper text | Omi | PC | Nov 3-11 | - | $0 |
| 9 | Release on Hugging Face (model card, FA segmentation, normalizer, scoring harness) | Omi (public) | HF | Nov 11 | - | $0 |
| 10 | Human verification of 300 references (optional, Gate 3: how much of the error is reference noise) | team | spreadsheet | Oct-Nov | ~3-4 h per person est. | $0 |
| 11 | Whisper-small variant (optional, size/speed trade-off) | you run 1 command | Modal | Nov | ~1 h est. | ~$2 est. |

**Your part: about 6 actions** (steps 0b, 3, 4, 5, 6, 7), each a single command or one Kaggle "Save & Run All".
The exact commands go into a Phase 5 runbook (like `docs/PHASE4_RUNBOOK.md`) when the code is ready.

## Money

| Account | Now | Phase 5 use |
|---|---|---|
| Omi's Modal | ~$11 left (reserve) | backup if the November run fails |
| Friend's Modal | ~$3.5 left, resets in November ($30) | steps 5-6 (~$11-14 est.), 11 |
| Kaggle (both) | free weekly GPU hours | steps 3, 7 |

Step 0b fits Omi's reserve (~$0.5-1). Step 5 at ~300 h (~$11-14 est.) does not fit the reserve alone, so it waits for
November's credit (or runs at 156 h on the reserve, ~$6-8 est., if the new corpora fail the checks in step 0).

## Decision gates
- **After step 0:** a corpus is used only if its license allows training and release, and it shares no speakers
  or sentences with our test sets; if none passes, M7 trains on the 156 h we have.
- **Licenses so far:** OpenSLR 53 = **CC BY-SA 4.0** (confirmed by Omi from the LICENSE file, 2026-10-12): training
  allowed; cite Kjartansson et al. 2018; anything that contains its audio or text (training lists, rewritten
  transcripts) is shared under CC BY-SA 4.0. **Decision at step 9 (Omi):** model weights under CC BY-SA 4.0
  (safe, recommended) or Apache-2.0 with attribution. Size and speaker/sentence overlap still to check.
- **After step 3:** if IndicConformer or a finalist model beats M1 full on the average, the "best open model"
  claim needs M7 to beat it; if not, report it as a baseline.
- **After step 7:** M7 must beat M5 (13.18) on the average and stay close to M1 full on DL Sprint eval (11.34);
  if it loses on DL Sprint by > 1 point, add a short DL Sprint-only second epoch (est. $2).
- **Step 10 (if done):** if more than ~20% of eval references are corrected, every model's WER is also reported
  against the verified references.

## Aiming for the top (added 2026-10-12)

Where we stand: first on DL Sprint media speech by ~7-10 points among open models; ~2 points behind Hishab on
FLEURS (16.44 vs 14.24); IndicConformer and the finalist models not measured yet (step 3). Same Whisper-medium
throughout; nothing here needs thousands of hours of training.

**Usability rule:** the released model is ONE Whisper-medium with greedy or light beam decoding, and every
candidate gets a speed number (real-time factor from `metrics.json`, T4). Ensembles and 12-best voting are
allowed only offline, as teachers for training labels (idea 4), never in the released system.

| # | Idea | Deployable? | Cost | How (you run) | Expected (est., unmeasured) |
|---|---|---|---|---|---|
| T1 | **Model soup**: average the weights of fine-tuned models (M1 + M5; all sources; WiSE-FT with zero-shot) | yes (one model, same speed) | $0 | Kaggle session D (below) | −0.5 to −1 on the average |
| T2 | **Cross-model ROVER** over the greedy outputs of ~10 fine-tuned models (members chosen on dev) | no (N x slower): teacher only | $0 | `PYTHONPATH=src ./Scripts/python.exe analysis/cross_model_rover.py` (CPU, minutes) | −0.5 to −1.5 |
| T3 | **Beam search** (5 beams) instead of greedy, with its speed cost measured | yes, if the slowdown is acceptable | $0 | Kaggle session D | small; speed ~2-4x slower |
| T4 | **Pseudo-labels**: transcribe unlabeled Bangla audio with the best teacher (T2 ensemble or M7), keep clips the models agree on (our agreement signal, AUROC 0.76-0.84), train M7 on labeled + pseudo-labeled data | yes (it is training data) | labeling $0 on Kaggle (~50 T4-h per 1,000 h audio); training ~$20-25 per +500 h | step 0 must find unlabeled audio with a usable license first | the lever that can close the FLEURS gap |
| T5 | **Corpus tag in the prompt** while training M7 (one model that writes each corpus's spelling convention) | yes | ~$0 extra | part of step 5 | −0.5 to −1 on the average |

**Kaggle session D (T1 + T3), after `bash runs/upload_code_to_hub.sh`:**
`GPU0 = "soup:m1_full+m5_merged,soup:m1_full+m1_full+m1_full+base"`,
`GPU1 = "beam:m1_full,soup:m1_full+m2_cv+m3_kathbath+m4_shrutilipi+m5_merged"` (~5-7 h est.; ~25 GB disk est.).
Winners are chosen on DL Sprint dev plus the average, then the same recipe is applied to M7 in November.

**Evidence for the FLEURS gap (2026-10-12, PROGRESS §62):** Hishab leads by 2.2 points; ~70% of it is on words that
occur in our training text (acoustics of read, formal speech), ~30% on words that never occur in it. Words never seen
in training text (10% of FLEURS) cost every model ~5 WER points, Hishab included: the pool where a lead "by a mile"
is possible.

| # | Idea | Targets | Deployable? | Cost |
|---|---|---|---|---|
| T6 | **More read, formal speech** (OpenSLR 53, IndicVoices; Kathbath already gave the best FLEURS row) | the seen-word part (+1.6) | yes | inside step 5 |
| T7 | **Synthetic speech for missing words**: an open Bangla TTS reads ~50-100 h of formal sentences rich in words absent from our training text; added to M7's training data | the unseen-word pool (~5 points, everyone) | yes (it is training data) | TTS on Kaggle $0; ~+$3-5 training est. |
| T8 | **Big-vocabulary LM in beam search** (shallow fusion; n-gram from a large Bangla text corpus) | unseen words at decoding time | yes, if beam search's speed cost is OK (T3 measures it) | $0 |
| - | Clean-test rule for T7/T8: drop every text sentence that overlaps FLEURS / Kathbath / CV / DL Sprint test sentences (n-gram check) | | | |

