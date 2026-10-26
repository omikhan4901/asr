# Runbook: the final model

Every step you run, in order. Git Bash in the project folder (`D:\Projects\ASR Paper Code`) unless it says
Kaggle. Times and costs marked *est.* are estimates. The last step writes `results/final_model_summary.txt`.

**What this measures:** M5-full (M5's recipe on all 156 h) and M5-300 (+ ~144 h of OpenSLR 53 read speech,
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
bash runs/upload_longform.sh
```
Expected end: `done: zips/longform-eval.zip is on the hub`. If it stops, run it again (finished files are kept).

### 2. Send the current code to the hub (~1 min, $0)
```bash
bash runs/upload_code_to_hub.sh
```
Expected end: `hub code/latest.txt = <12 characters> (OK)`. Run it again whenever the code changes.

---

## October (new Modal account, fresh $30): train M5-full and M5-300 now

The new account's volume starts empty. M5-full needs only data that is on the hub (N5). M5-300 also needs the
OpenSLR zip, which is not on the hub, so N6b rebuilds it on the new volume (same script, seed 0). Total est. ~$26
(caps $12.5 + $15.5). The friend's November credit then stays free as the backup. If N6b is skipped, M5-300 runs
on the friend's account on Nov 1 (below).

| # | Do | Time | Cost |
|---|---|---|---|
| N1 | modal.com: sign up, add the card, check Settings -> Usage & Billing shows the $30 free credit | 5 min | $0 |
| N2 | `./Scripts/modal.exe token new --profile newacct` (approve in the browser), then `./Scripts/modal.exe profile activate newacct` and `./Scripts/modal.exe profile current` (must print `newacct`) | 2 min | $0 |
| N3 | modal.com/secrets -> Create new secret -> Custom: name `huggingface`, key `HF_TOKEN`, value = a write token from huggingface.co/settings/tokens | 2 min | $0 |
| N4 | `./Scripts/python.exe pipeline/modal_launch.py deploy` | 2 min | $0 |
| N5 | `PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::from_hf --patterns "zips/splits_fa-train.zip,zips/splits_fa-dev.zip,zips/cv-train.zip,zips/kathbath-train.zip,zips/shrutilipi-train.zip,manifests/*"` (keep the window open; if it stops, run it again) | ~20-40 min est. | ~$0.2 est. |
| N6 | `./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_full --source m5full --budget 12.5`, then `status` until `exit code 0` | ~5 h est. | ~$11 est. |
| N6b | (at the same time as N6) `./Scripts/python.exe pipeline/modal_launch.py prep --sources openslr53`, `status` until the `prep` line says `exit code 0`, then `./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_300 --source m5_300 --budget 15.5` | prep ~1-2 h est., training ~6.5 h est. | ~$0.5 + ~$14 est. |
| N7 | `PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m5_full` (and `--run m5_300` when it is done) | ~5 min each | ~$0.05 each |
| N8 | `./Scripts/modal.exe profile activate mehboobehsankhan` | - | - |

After N7, M5-full can be scored on Kaggle right away (step 9a, `GPU0` line).

## November 1: M5-300 on the friend's Modal (the data is already on their volume)

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

### 5. DONE (2026-10-12/13): OpenSLR 53 and both training lists are on the friend's volume
OpenSLR 53: 22,786 joined samples, 144.1 h, 317 speakers, 24 test sentences dropped; `m5-full` 73,879 rows,
`m5-300` 96,665 rows (checked read-only 2026-10-19). Skip this step. For reference, the command was:
```bash
./Scripts/python.exe pipeline/modal_launch.py prep --sources openslr53
./Scripts/python.exe pipeline/modal_launch.py status
```
Repeat `status` until the newest `prep` line says `finished, exit code 0`. It downloads OpenSLR 53 (~14.7 GB) inside
Modal, joins the clips, drops the 11 sentences that are also in Common Voice test, and builds `m5-full` (156 h) and
`m5-300` (~300 h). If it fails with "No space left on device", stop and fix the prep step before going on.

### 6. Train M5-300 (~6.5 h est.; stops itself at its cap). M5-full only if N6 above was not done
```bash
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

Same notebook as before (`notebooks/kaggle_jobs.ipynb`), GPU T4 x2, Internet on, secret `HF_TOKEN`. Set the first
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
If it ends early (time limit), resume as in `docs/RUNBOOK_finetuning.md` 2.4 (add its output as input, set `RESUME_FROM`).

### 10. Put the outputs in place
For each session: **Output** tab -> download `kaggle_outputs.zip` -> unzip into its own folder, e.g.
`data/kaggle_final_model/scores/` and `data/kaggle_final_model/stack/` (each must contain `outputs/`).

### 11. Final analysis (~20-40 min, CPU)
```bash
bash runs/final_model_finish.sh
```
It writes `results/final_model_summary.txt`, the source for the results, README, figures and the faculty table.

---

## Pre-registered (written before any final-model score)
- M5-full: DL Sprint eval <= 12.0, every other set within 0.5 of its specialist, average <= 12.5.
- M5-300: average significantly below M5-full's; otherwise more hours of this kind do not help.
- ROVER target = M5-300 (fixed now, not chosen by test scores).

## Not verified yet
- The Modal steps (5-7) with the new data: the joining and the list builder were tested on the PC (80 real
  OpenSLR clips; the M5 list is reproduced exactly); the full download and the Modal disk size were not.
- The Kaggle jobs `longform:`, `stack:<set>@<run>`, `soup:`, `beam:`: command construction checked on the PC,
  long-form evaluation run on 40 s of one recording (CPU); not run on Kaggle yet.
