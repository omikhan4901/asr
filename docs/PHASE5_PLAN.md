# Phase 5 plan: the released model, whole-recording evaluation, release

Phase 4 answered the method questions. Phase 5 builds the one model we release and finishes the paper.
Freeze: **Nov 12**; report: **Dec 1**. Times and costs marked *est.* are estimates.

## What Phase 4 decided for Phase 5

| Question | Answer (Phase 4) | Consequence for Phase 5 |
|---|---|---|
| Which training method? | full fine-tuning is best (11.34% eval; LoRA 11.76, decoder 12.99) | full fine-tuning |
| Which data? | each corpus wins its own test set; 10 h x 4 has the best average (13.18%) | train on **all four corpora together** (~156 h) |
| Harmonize the spelling? | no gain on DL Sprint, small loss on Common Voice (Gate 4) | **no** harmonization |
| Longer training / higher lr? | 4 epochs at 2e-5 over-fits (out-of-domain +1.4 to +2.3) | keep **1e-5, 2 epochs**; choose with a **mixed dev set** |
| Rescoring on top? | ROVER gives −0.45 after fine-tuning | ROVER on the final model's 12-best |

## Steps

| # | Step | Who | Where | When | Time | Cost |
|---|---|---|---|---|---|---|
| 1 | Merged training list (DL Sprint 59.6 h + CV 34.4 h + Kathbath 33.9 h + Shrutilipi 28.1 h), mixed dev set (DL Sprint dev + CV dev + FLEURS dev + held-out Kathbath train speakers), `--source all` | Omi | PC (CPU) | Oct | - | $0 |
| 2 | Whole-recording evaluation code (VAD cuts the raw recording, the model transcribes, the text is joined, WER per recording on the 22 eval recordings) | Omi | PC | Oct | - | $0 |
| 3 | Kaggle job for baselines: IndicConformer + the released DL Sprint finalist models on the 4 test sets; whole-recording eval of zero-shot and M1 full | Omi, **1 Kaggle session** | Kaggle | Oct | ~6-9 h est. | $0 |
| 4 | Upload the merged list to the hub | **you run 1 command** | PC | Oct 31 | ~10 min | $0 |
| 5 | Train the released model (M7) | **you run 1 command** | Modal (November credit) | Nov 1 | ~3 h est. | ~$6-8 est. |
| 6 | Send M7 to the hub (`to_hf --run m7_all`) | **you run 1 command** | Modal | Nov 1 | ~10 min | ~$0.05 |
| 7 | Score M7 + its 12-best for ROVER + whole-recording eval | **you start** (1-2 Kaggle sessions) | Kaggle | Nov 2-3 | ~10 h est. | $0 |
| 8 | Analysis, tables, figures, README, paper text | Omi | PC | Nov 3-11 | - | $0 |
| 9 | Release on Hugging Face (model card, FA segmentation, normalizer, scoring harness) | Omi (public) | HF | Nov 11 | - | $0 |
| 10 | Human verification of 300 references (optional, Gate 3: how much of the error is reference noise) | team | spreadsheet | Oct-Nov | ~3-4 h per person est. | $0 |
| 11 | Whisper-small variant (optional, size/speed trade-off) | you run 1 command | Modal | Nov | ~1 h est. | ~$2 est. |

**Your part: about 5 actions** (steps 3, 4, 5, 6, 7), each a single command or one Kaggle "Save & Run All".
The exact commands go into a Phase 5 runbook (like `docs/PHASE4_RUNBOOK.md`) when the code is ready.

## Money

| Account | Now | Phase 5 use |
|---|---|---|
| Omi's Modal | ~$11 left (reserve) | backup if the November run fails |
| Friend's Modal | ~$3.5 left, resets in November | steps 5-6 (and 11) |
| Kaggle (both) | free weekly GPU hours | steps 3, 7 |

Option (your decision): step 5 can start before November on Omi's ~$11 reserve (est. $6-8), which leaves
more time before the freeze but no backup until November.

## Decision gates
- **After step 3:** if IndicConformer or a finalist model beats M1 full on the average, the "best open model"
  claim needs M7 to beat it; if not, report it as a baseline.
- **After step 7:** M7 must beat M5 (13.18) on the average and stay close to M1 full on DL Sprint eval (11.34);
  if it loses on DL Sprint by > 1 point, add a short DL Sprint-only second epoch (est. $2).
- **Step 10 (if done):** if more than ~20% of eval references are corrected, every model's WER is also reported
  against the verified references.
