# Bangla subtitle demo

Upload a Bangla video or audio file (up to 15 minutes) and download an `.srt` subtitle file. It uses the released
model ([OmiKhan/bangla-asr-whisper-medium](https://huggingface.co/OmiKhan/bangla-asr-whisper-medium)) and the same
path as the whole-recording evaluation of the paper (`pipeline/eval_longform.py`): Silero voice detection cuts the
audio into pieces of up to 25 s and each piece is decoded greedily. The forced aligner that built the evaluation split
(`src/bangla_asr/preprocessing/forced_align.py`, MMS_FA) then times every word, and the words are grouped into
subtitles of at most two lines of 42 characters and 7 seconds, cut at pauses and sentence ends (`backend/srt_utils.py`).
The text is exactly what the model wrote; alignment only sets the times.

| Part | Where | What |
|---|---|---|
| `frontend/` | Vercel (`asr-phi.vercel.app`) or the local server | One page, English and Bangla, no build step |
| `backend/app.py` | Modal (A10G GPU, scales to zero) | `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/download` |
| `backend/local_server.py` | this PC | The same API, and it serves the frontend too |
| `backend/pipeline.py` | both | Audio extraction, voice detection, transcription, word timing, SRT |

Uploaded videos are deleted as soon as their subtitles are made; only the `.srt` is kept.

## Run it on your own GPU

Needs an NVIDIA GPU with about 4 GB free (it uses 3.2 GB), `python3` and `ffmpeg`. From the repository root:

```bash
web/run_local.sh
```

The first run builds `web/.venv-gpu` with CUDA PyTorch (about 3 GB, 5 to 10 minutes). It then prints the GPU it
found and starts the server. Open http://127.0.0.1:8000 when the log says `Uvicorn running`. On an RTX 3060 a 47 s clip takes
3 s from upload to `.srt` (the first file also loads the models;
the very first one downloads the 1.2 GB word aligner, about 3 minutes). Stop it with Ctrl+C. Without a GPU it still runs, slowly, on the CPU.

The model comes from `data/release/model` if that folder exists, otherwise from Hugging Face (downloaded once).
Set `BANGLA_ASR_MODEL` to use another checkpoint.

## Deploy

Backend, from the repository root (Modal CLI logged in to the account that owns the app):

```bash
.venv/bin/modal deploy web/backend/app.py
```

Frontend: `frontend/` is a static folder. On any public host (Vercel, a custom domain) the page calls the Modal URL in
`script.js` (`MODAL_API`); on localhost it calls its own origin. Set `window.ASR_API_BASE` before `script.js` loads to point it elsewhere.
