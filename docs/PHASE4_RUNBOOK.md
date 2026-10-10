# Phase 4 runbook: finishing Phase 4 without the home GPU

Plain step-by-step guide. Do the parts in order; each step says where you do it, what to type or
click, how long it takes, what it costs, and what you should see. Nothing here uses the PC's GPU.

**The idea in one picture**

```
 Omi's PC ──(Common Voice train, 0.75 GB, once)──────────────┐
                                                             ▼
 Omi's Modal volume ──(to_hf: test audio, models, scores, code)──► Hugging Face private repo
                                                             │     OmiKhan/bangla-asr-hub
                       ┌─────────────────────────────────────┤
                       ▼                                     ▼
          Kaggle (Omi + friend, free T4s)          Friend's Modal (A100, $30 credit)
          scoring, T4 check, 12-best for stacking  matrix training M2-M5
                       │                                     │
                       └──► phase4_outputs.zip ──► Omi's PC ◄┘ (models go back to the hub, then Kaggle scores them)
```

Hugging Face is the one place everything meets: fast downloads for Kaggle and for the friend's
Modal, so nothing big has to go up from the PC except Common Voice once.

**Money (estimates; the real number is on the Modal dashboard).** The budget caps now add $0.15 for
every container start, which the old estimates missed.

| Account | What | Est. cost |
|---|---|---|
| Omi's Modal | `to_hf` (CPU only, ~15-25 min incl. first image build) | ~$0.10-0.25 |
| Friend's Modal | `from_hf` + matrix data prep (CPU only) | ~$0.3-0.6 |
| Friend's Modal | M2 CV, M3 Kathbath, M4 Shrutilipi, M5 merged (A100, full fine-tune, ~34 h of audio each) | ~$2.5-3 each, ~$11-13 total |
| Friend's Modal | learning-rate check (2 runs) + 4-epoch run (3.7b) | ~$10 total |
| Kaggle (both) | T4 check, LoRA leftovers, stacking 12-best, matrix scoring | $0 (uses the weekly 30 GPU-hours) |

Omi's leftover credit (~$11) stays as the reserve for Phase 5.

---

## Part 0. Tokens and secrets (once, ~15 min, $0)

**Why:** the private repo `OmiKhan/bangla-asr-hub` is the shared folder between your Modal, your friend's Modal,
both Kaggle accounts and your PC. Every program that reads or writes it needs a token (a password for programs;
never paste one into chat). Three tokens, each with only the rights it needs:

| Token | Token type | Can do | Goes into |
|---|---|---|---|
| `modal-hub` | **Write** | read + write all your repos, read gated datasets you accepted | your Modal secret (0.2) + your PC login (0.3) |
| `kaggle-read` | **Read** | read only | the Kaggle notebooks, yours and your friend's (Part 2) |
| `friend-modal` | **Fine-grained** | read + write only `bangla-asr-hub`, read gated datasets | your friend's Modal secret (Part 3) |

A token is shown only once. Lost or leaked: tokens page -> **⋮ -> Delete**, make a new one, update where it was used.

### 0.1 Write token (for your Modal account and your PC)
1. Open https://huggingface.co/settings/tokens (logged in as **OmiKhan**).
2. Click **+ Create new token**.
3. **Token type:** click **Write** in the top row (Fine-grained · Read · Write). This is the token type, not the
   "Write" button in the *Presets* row below it. With Write selected there are no permission boxes to tick.
4. **Token name:** `modal-hub`. Click **Create token**.
5. Copy the token (starts with `hf_`). Keep the tab open for the next two steps.

### 0.2 Put the write token in your Modal account
1. Open https://modal.com/secrets (logged in as yourself).
2. Click **Create new secret** -> **Custom**.
3. Name: `huggingface` (exactly this, lower case).
4. Key: `HF_TOKEN`  Value: paste the write token. Click **Create**.

### 0.3 Log the PC in with the write token
In Git Bash in the project folder (`D:\Projects\ASR Paper Code`):
```bash
./Scripts/hf.exe auth login
```
Paste the write token when asked (nothing shows while you paste; that is normal), press Enter.
If it asks "Add token as git credential?" type `n`. Check:
```bash
./Scripts/hf.exe auth whoami
```
It must print `user=OmiKhan`.

### 0.4 Read token (for Kaggle; safe to give your friend)
1. Same page, **+ Create new token** -> **Token type: Read** (top row) -> name `kaggle-read` -> **Create token**.
2. Save it somewhere private (a notes app). It can read your private repos but cannot change anything.

### 0.5 Token for your friend's Modal (do this AFTER Part 1.2: the repo must exist first)
1. **+ Create new token** -> **Token type: Fine-grained** -> **Presets: Custom**. Name `friend-modal`.
2. Under **Repositories permissions**, type `bangla-asr-hub` in the search box, pick **OmiKhan/bangla-asr-hub**,
   and tick **Read access to contents** and **Write access to contents**.
3. Under **User permissions -> Repositories**, tick **Read access to contents of all public gated repos you can access**
   (needed for Kathbath and Shrutilipi).
4. Leave everything else unticked. **Create token**, save it privately. It goes into the friend's Modal in Part 3.3.

---

## Part 1. Your PC and your Modal (~30-60 min, mostly waiting, ~$0.25)

Everything below is in Git Bash in the project folder. None of it uses the GPU.

### 1.1 Get the latest code
```bash
git pull
```

### 1.2 Upload Common Voice train to the hub (0.75 GB)
This creates the private repo on the first run.
```bash
./Scripts/hf.exe upload OmiKhan/bangla-asr-hub data/matrix/hub_upload . --repo-type dataset --private
```
Time: depends on your upload speed (0.75 GB: ~5 min at 20 Mbit/s up, ~25 min at 4 Mbit/s).
Expected end: a line with a link `https://huggingface.co/datasets/OmiKhan/bangla-asr-hub/...`.
If it stops halfway, run the same command again: it skips what is already uploaded.

### 1.3 Copy everything from your Modal volume to the hub
First a dry run (lists what would go, sends nothing, ~2 min, a few cents because the small CPU image is built once):
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --dry
```
You should see lines like `zips -> OmiKhan/bangla-asr-hub/zips: 6 files, 6.6 GB`, `checkpoints/m1_full/final -> ... models/m1_full: 7 files, 3.06 GB`, then `done`.
Then the real copy (~10-20 min; keep the window open until it prints `done`):
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf
```
Check on the website: https://huggingface.co/datasets/OmiKhan/bangla-asr-hub/tree/main should now show
the folders `code`, `manifests`, `models`, `results`, `zips`.

**If the window closes by accident:** run the same command again (uploads that finished are not repeated).

---

## Part 2. Kaggle (no PC needed while it runs, $0)

Two sessions, one on each Kaggle account, both at the same time. Session A first if you only start one.

### 2.1 Set up a notebook (do this once per Kaggle account)
1. Go to https://www.kaggle.com/code and click **+ New Notebook**.
2. Menu **File -> Import Notebook**, choose the file `notebooks/kaggle_phase4.ipynb` from the project folder, **Import**.
   (Your friend needs this file too: send it to them, it is small and contains no secrets.)
3. Right side panel, **Session options**:
   - **Accelerator: GPU T4 x2**
   - **Internet: on** (needs a phone-verified account)
4. Menu **Add-ons -> Secrets -> + Add Secret**: Label `HF_TOKEN`, Value = the **read** token from 0.4. Make sure its
   switch is on for this notebook.

### 2.2 Choose the jobs
In the first code cell, keep one preset and leave the others commented out (a `#` at the start of a line turns it off):

| Session | GPU0 | GPU1 | Est. time |
|---|---|---|---|
| **A** (friend's account) | `t4check` | `lora` | ~1.5-2.5 h |
| **B** (your account) | `stack:eval` | `stack:dev,stack:fleurs,stack:kathbath` | ~6-8 h (est.) |

What they do:
- `t4check`: scores the zero-shot model on dev, FLEURS and Kathbath on the T4 and compares every sentence with the A100 results. Kaggle numbers are trusted only if these match (A100 vs 3060 matched on 96-99% of sentences, WER within 0.02).
- `lora`: finishes M1 LoRA on Kathbath and Common Voice (the Modal credit ran out there).
- `stack:<set>`: the input for stacking (Phase 4.7): 12 candidates per clip from M1 full, each scored exactly, plus M1 full's greedy output.

### 2.3 Start it
1. Top right **Save Version** -> **Save & Run All (Commit)** -> **Save**.
2. You can close the browser and turn the PC off. Kaggle runs it on its own (up to 12 h).
3. To watch: https://www.kaggle.com/code -> your notebook -> the version -> **Logs**. A status line appears every 5 minutes.

### 2.4 When it is finished
1. Open the notebook version -> **Output** tab -> download `phase4_outputs.zip`.
2. Put it in the project folder as `data/kaggle_out/session_A.zip` (or `session_B.zip`); create the folder if needed.
3. Start the analysis session ("Kaggle session A outputs are in data/kaggle_out"): the outputs are unpacked, checked and analysed there.

**If a session stops before the end** (timeout, error): open its **Output**, click **New Notebook** / **Add to notebook**
(adds the old output as an input), set `RESUME_FROM = "/kaggle/input/<that output's folder>/phase4"` in the first cell,
and run again with the same preset. Finished sentences are skipped.

**If cell 3 fails with "No GPU"**: Session options -> Accelerator -> GPU T4 x2, then **Run All** again.
**If it fails with 401 / 403 from Hugging Face**: the `HF_TOKEN` secret is missing, off, or mistyped (Add-ons -> Secrets).

---

## Part 3. Your friend's Modal account: matrix training (from tomorrow, run on YOUR PC)

Your friend does not need the code. They log in to their Modal account on your PC once; from then on
you switch between the two accounts with one command.

### 3.1 Add the friend's account to your PC (~5 min, friend present or on a call)
First check: you already have a second Modal profile called `asr499` (workspace `499asr`). If that is
your friend's account, skip to 3.2 and use `asr499` instead of `friend` below.
```bash
./Scripts/modal.exe token new --profile friend
```
A browser window opens: your friend logs into **their** Modal account there and clicks **Authorize**.

### 3.2 Switch the PC to the friend's account
```bash
./Scripts/modal.exe profile activate friend
./Scripts/modal.exe profile current
```
The second command must print `friend`. **Everything from here to 3.7 runs on the friend's credit.**

### 3.3 Safety settings in the friend's Modal (friend clicks, ~3 min)
1. https://modal.com/settings -> **Usage / Billing**: set a **workspace budget of $30** and do not add a card,
   so nothing can be charged beyond the free credit.
2. https://modal.com/secrets -> **Create new secret -> Custom**, name `huggingface`, key `HF_TOKEN`,
   value = the fine-grained token from 0.5.

### 3.4 Deploy the code to the friend's account (~2 min, $0)
```bash
./Scripts/python.exe pipeline/modal_launch.py deploy
```
Expected: `deployed commit xxxxxxx to Modal account 'friend'`.

### 3.5 Fill the friend's volume from the hub (~5-10 min, ~$0.05)
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::from_hf
```
Expected: a list of files (`zips/splits_fa-train.zip`, `zips/splits_fa-dev.zip`, `zips/cv-train.zip`, manifests) and `done`.

### 3.6 Prepare Kathbath and Shrutilipi (~40-80 min est., ~$0.2-0.4; runs on its own)
```bash
./Scripts/python.exe pipeline/modal_launch.py prep
./Scripts/python.exe pipeline/modal_launch.py status
```
`status` says `running` until it is done, then `finished, exit code 0`. You may close everything meanwhile.
It takes 34 h from each corpus (same size as Common Voice train, so the four matrix rows are comparable),
removes the Kathbath speakers of our Kathbath test set, applies the same text rules as DL Sprint, and
builds the M5 merged list (10 h from each of DL Sprint, CV, Kathbath, Shrutilipi).

### 3.7 Train the matrix (4 jobs, in parallel, ~1-1.5 h each, ~$2.5-3 each)
Start these four; each stops by itself at its budget cap:
```bash
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m2_cv --source cv --budget 3.5
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m3_kathbath --source kathbath --budget 3.5
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m4_shrutilipi --source shrutilipi --budget 3.5
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_merged --source merged --budget 3.5
```
Check with `./Scripts/python.exe pipeline/modal_launch.py status`. Live logs: modal.com -> Apps -> `bangla-asr-finetune` -> the function's calls.
`exit code 0` = finished; `3` = stopped at the cap with a saved `partial/` model (keep the last 20 log lines for the next analysis session).

### 3.7b Learning rate and epochs check (same account, start together with 3.7; ~$10 est. in total)
The current recipe (learning rate 1e-5, 2 epochs) was a first guess. These runs use the same DL Sprint data as M1
full and change one thing each, so they compare directly with M1 full (training-time dev WER 16.73%).

Step 1, at the same time as the matrix (~1 h each, ~$2.4-2.6 each):
```bash
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m1_full_lr5e-6 --extra "--lr 5e-6" --budget 3
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m1_full_lr2e-5 --extra "--lr 2e-5" --budget 3
```
Step 2, when both say `finished, exit code 0` in `status`: start the analysis session ("learning-rate runs finished").
It reads their dev results and names the winning learning rate (lowest dev WER among 5e-6, 1e-5, 2e-5; never a test set).
Then (~1.5-2 h, ~$4.5 est.), with `<LR>` replaced by the winner:
```bash
./Scripts/python.exe pipeline/modal_launch.py train --method full --run m1_full_4ep --extra "--epochs 4 --lr <LR> --keep_best" --budget 5
```
`--keep_best` also keeps the model from the epoch with the lowest dev WER as `best/`. If dev WER is still clearly
falling at epoch 4, a 6-8 epoch run is the next step (~$1 per extra epoch); if it has flattened, 4 is enough.
Step 3, send both 4-epoch models to the hub and score them on Kaggle like the matrix models:
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m1_full_4ep
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m1_full_4ep --which best
```
Kaggle preset: `GPU0 = "score:m1_full_4ep_best"` (and `score:m1_full_4ep` if the best epoch was not the last).

### 3.8 Send the trained models to the hub (~5 min each, ~$0.02 each)
```bash
PYTHONIOENCODING=utf-8 ./Scripts/modal.exe run pipeline/modal_finetune.py::to_hf --run m2_cv
```
(then the same with `m3_kathbath`, `m4_shrutilipi`, `m5_merged`).

### 3.9 Switch the PC back to your own account
```bash
./Scripts/modal.exe profile activate mehboobehsankhan
```

### 3.10 Score the matrix models on Kaggle
Same notebook as Part 2, presets:
`GPU0 = "score:m2_cv,score:m4_shrutilipi"` and `GPU1 = "score:m3_kathbath,score:m5_merged"` (~6-9 h est.; split
over two sessions if needed). Download and hand over as in 2.4.

---

## What happens with the results
- Session A: T4 vs A100 per sentence; if they match, LoRA's Kathbath/CV rows complete Table "Phase 4 M1".
- Session B: stacking (Phase 3 expansion + LM-MBR on the fine-tuned model's 12-best) and E11 (anatomy after fine-tuning).
- Matrix: the train x test table (M1-M5), then M6 (harmonized spelling) is designed.

## Things that are NOT verified yet (first real run will tell)
- Anything on Kaggle itself (the job script was tested on this PC on CPU with 2 sentences per set, the notebook cells were not run on Kaggle).
- The Modal CPU functions (`to_hf`, `from_hf`, `prep_matrix`) have not run on Modal yet; the data preparation itself
  was tested on this PC (Kathbath and Shrutilipi samples, full Common Voice).
- All times and dollar amounts marked est.
