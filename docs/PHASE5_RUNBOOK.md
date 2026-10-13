# Phase 5 runbook: from now to the final numbers

Every step you run, in order. Git Bash in the project folder (`D:\Projects\ASR Paper Code`) unless it says
Kaggle. Times and costs marked *est.* are estimates. The last step writes `results/phase5_summary.txt`.

**What Phase 5 measures:** M5-full (M5's recipe on all 156 h) and M5-300 (+ ~144 h of OpenSLR 53 read speech,
short clips joined into ~25 s samples), each against M5 (13.18 average); ROVER on M5-300 (fixed in advance);
whole-recording WER (raw recordings cut only by voice detection).

| When | Steps | Where | Cost |
|---|---|---|---|
| now | 1-2 | your PC | $0 |
| Nov 1 | 3-8 | your PC, friend's Modal | ~$25-28 est. |
| Nov 1-2 | 9-11 | Kaggle (both accounts) + your PC | $0 |

---

## Now (no Modal, no Kaggle)

### 1. Send the 22 raw eval recordings to the hub (~10-20 min, $0)
```bash
bash runs/phase5_upload_longform.sh
```
Expected end: `done: zips/longform-eval.zip is on the hub`. If it stops, run it again (finished files are kept).

### 2. Send the current code to the hub (~1 min, $0)
```bash
bash runs/upload_code_to_hub.sh
```
Expected end: `hub code/latest.txt = <12 characters> (OK)`. Run it again whenever the code changes.

---

## November 1: training on the friend's Modal (the data is already on their volume)

### 3. Switch to the friend's account
```bash
./Scripts/modal.exe profile activate friend
./Scripts/modal.exe profile current
```
Must print `friend`. Everything until step 8 bills the friend's credit ($30 resets in November).

### 4. Deploy the code (~2 min, $0)
```bash
./Scripts/python.exe pipeline/modal_launch.py deploy
```

### 5. Prepare OpenSLR 53 and the two training lists (~1-2 h est., ~$0.3-0.6 est.; runs on its own)
```bash
./Scripts/python.exe pipeline/modal_launch.py prep --sources openslr53
./Scripts/python.exe pipeline/modal_launch.py status
```
Repeat `status` until the newest `prep` line says `finished, exit code 0`. It downloads OpenSLR 53 (~14.7 GB) inside
Modal, joins the clips, drops the 11 sentences that are also in Common Voice test, and builds `m5-full` (156 h) and
`m5-300` (~300 h). If it fails with "No space left on device", stop and fix the prep step before going on.

### 6. Train both models (in parallel; ~5 h and ~6.5 h est.; each stops itself at its cap)
```bash
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_full --source m5full --budget 12.5
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_300 --source m5_300 --budget 15.5
```
Check with `status` (you may close everything meanwhile). `exit code 0` = done. `exit code 3` = stopped at the cap
with a saved `partial/` model: check it before doing anything else. Est. cost ~$11 + ~$14 (from M1's measured
$2.22 for 15,325 clips; cost follows the number of clips).

### 7. Send both models to the hub (~5 min each, ~$0.05 each)
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m5_full
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m5_300
```

### 8. Switch back to your account
```bash
./Scripts/modal.exe profile activate mehboobehsankhan
```

---

## Kaggle: scoring (two sessions, one per account, at the same time; $0)

Same notebook as Phase 4 (`notebooks/kaggle_phase4.ipynb`), GPU T4 x2, Internet on, secret `HF_TOKEN`. Set the first
cell, then **Save Version -> Save & Run All (Commit)**. Never press Stop (a stopped run loses its output).

### 9a. Your account: scores + whole recordings (~4-5 h est.)
```python
GPU0 = "score:m5_full,longform:base,longform:m1_full"
GPU1 = "score:m5_300,longform:m5_300"
RESUME_FROM = ""
```

### 9b. Friend's account: 12-best of M5-300 for ROVER (~10 h est.; Kaggle's limit is 12 h)
```python
GPU0 = "stack:eval@m5_300"
GPU1 = "stack:dev@m5_300,stack:fleurs@m5_300,stack:kathbath@m5_300"
RESUME_FROM = ""
```
If it ends early (time limit), resume as in `docs/PHASE4_RUNBOOK.md` 2.4 (add its output as input, set `RESUME_FROM`).

### 10. Put the outputs in place
For each session: **Output** tab -> download `phase4_outputs.zip` -> unzip into its own folder, e.g.
`data/phase5/scores/` and `data/phase5/stack/` (each must contain `phase4/`).

### 11. Final analysis (~20-40 min, CPU)
```bash
bash runs/phase5_finish.sh
```
It writes `results/phase5_summary.txt`, the source for the results, README, figures and the faculty table.

---

## Pre-registered (written before any Phase 5 score)
- M5-full: DL Sprint eval <= 12.0, every other set within 0.5 of its specialist, average <= 12.5.
- M5-300: average significantly below M5-full's; otherwise more hours of this kind do not help.
- ROVER target = M5-300 (fixed now, not chosen by test scores).

## Not verified yet
- The Modal steps (5-7) with the new data: the joining and the list builder were tested on the PC (80 real
  OpenSLR clips; the M5 list is reproduced exactly); the full download and the Modal disk size were not.
- The Kaggle jobs `longform:`, `stack:<set>@<run>`, `soup:`, `beam:`: command construction checked on the PC,
  long-form evaluation run on 40 s of one recording (CPU); not run on Kaggle yet.
