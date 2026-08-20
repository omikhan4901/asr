"""
Local-only test server — same endpoints/contract as the Modal app.py, but
runs entirely on your machine (no Modal Volume, no Dict, no spawn). Reuses
SubtitleGenerator from pipeline.py unchanged, so a successful local run
here is a real test of the actual pipeline logic, not a mock.

Run:
    cd web/backend
    pip install fastapi uvicorn python-multipart torch transformers accelerate librosa soundfile jiwer numpy silero-vad
    uvicorn local_server:app --reload --port 8000

Requires ffmpeg on your system PATH (same as the Modal image installs via apt).
Requires CUDA — this will raise clearly if torch.cuda.is_available() is False,
same as every other script in this project.
"""

import os
import uuid
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline import SubtitleGenerator

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

DATA_DIR = Path("local_data")
(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)

# In-memory job store — fine for local single-process testing; this is the
# local stand-in for Modal's job_store Dict.
jobs: dict[str, dict] = {}

# Loaded once at server startup (same as @modal.enter() does per-container).
print("Loading model + VAD (this happens once, may take a minute)...")
generator = SubtitleGenerator()
print("✅ Ready.")


def _update_job(job_id: str, **fields):
    jobs[job_id].update(fields)


def _run_job(job_id: str, video_path: str):
    def progress_cb(stage, current=None, total=None):
        fields = {"status": "processing", "stage": stage}
        if current is not None and total is not None:
            fields["progress"] = {"current": current, "total": total}
        _update_job(job_id, **fields)

    try:
        _update_job(job_id, status="processing", stage="audio")
        result = generator.process_video(video_path, progress_cb=progress_cb)

        srt_path = DATA_DIR / "outputs" / f"{job_id}.srt"
        srt_path.write_text(result["srt_text"], encoding="utf-8")

        _update_job(
            job_id,
            status="done",
            stage="srt",
            result={
                "n_segments": result["n_segments"],
                "duration_seconds": result["duration_seconds"],
                "srt_preview": result["srt_preview"],
                "srt_path": str(srt_path),
            },
        )
    except Exception as e:
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(e))


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    video_path = DATA_DIR / "uploads" / f"{job_id}_{file.filename}"

    contents = await file.read()
    video_path.write_bytes(contents)

    jobs[job_id] = {"status": "queued", "stage": "upload", "result": None, "error": None}

    # Background thread instead of Modal's .spawn() — same effect locally:
    # the HTTP response returns immediately, processing continues async.
    thread = threading.Thread(target=_run_job, args=(job_id, str(video_path)), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/download")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Result not ready")

    srt_path = Path(job["result"]["srt_path"])
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="File missing")

    return Response(
        content=srt_path.read_text(encoding="utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
    )