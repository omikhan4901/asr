"""
Local FastAPI development server matching the Modal web contract.
Implements thread-safe GPU locking and streaming upload handlers.
"""

import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

# Ensure project root and src are on python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from web.backend.pipeline import SubtitleGenerator
except ImportError:
    from pipeline import SubtitleGenerator

LOCAL_DATA_DIR = PROJECT_ROOT / "data/web_local"
UPLOADS_DIR = LOCAL_DATA_DIR / "uploads"
SRT_DIR = LOCAL_DATA_DIR / "srts"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SRT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 300
MAX_VIDEO_DURATION_S = 15 * 60
gpu_lock = threading.Lock()
jobs: Dict[str, Dict] = {}

app = FastAPI(title="Bangla ASR Subtitle Generator (Local Dev)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_generator: SubtitleGenerator = None


def get_generator() -> SubtitleGenerator:
    global _generator
    if _generator is None:
        _generator = SubtitleGenerator(max_duration_s=MAX_VIDEO_DURATION_S)
    return _generator


def _run_job(job_id: str, video_path: str):
    def progress_cb(stage: str, current: int, total: int):
        jobs[job_id]["stage"] = stage
        jobs[job_id]["progress"] = {
            "stage": stage,
            "current": current,
            "total": total,
            "percent": int(current / max(1, total) * 100),
        }

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["stage"] = "queued"

        with gpu_lock:
            generator = get_generator()
            result = generator.process_video(video_path, progress_cb=progress_cb)

        srt_path = SRT_DIR / f"{job_id}.srt"
        srt_path.write_text(result["srt_content"], encoding="utf-8")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "done"
        jobs[job_id]["result"] = {
            "num_chunks": result["num_chunks"],
            "total_speech_s": result["total_speech_s"],
            "cues": result["cues"],
            "srt_path": str(srt_path),
        }
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        print(f"[Job {job_id}] Failed: {exc}")
    finally:
        Path(video_path).unlink(missing_ok=True)  # the upload is only needed to make the subtitles


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename or "upload").name
    video_path = UPLOADS_DIR / f"{job_id}_{safe_name}"

    # Stream upload to disk in 1MB chunks to prevent memory bloat
    total_bytes = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    with open(video_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
            f_out.write(chunk)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "upload",
        "progress": {"stage": "upload", "current": 1, "total": 1, "percent": 100},
        "created_at": time.time(),
    }

    thread = threading.Thread(target=_run_job, args=(job_id, str(video_path)), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/download")
async def download_srt(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed")

    srt_path = Path(job["result"]["srt_path"])
    if not srt_path.exists():
        raise HTTPException(status_code=500, detail="SRT file missing")

    content = srt_path.read_text(encoding="utf-8")
    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
    )


# Mount static frontend directory so http://127.0.0.1:8000 serves the UI directly
FRONTEND_DIR = PROJECT_ROOT / "web/frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
