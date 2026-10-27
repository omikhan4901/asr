"""
Modal-deployed FastAPI application and GPU worker for the Bangla Subtitle Generator.
"""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import modal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("bangla-subtitle-generator")

def download_models():
    """Bake the ASR model and the word aligner into the image, so a cold start downloads nothing."""
    import torchaudio
    from huggingface_hub import snapshot_download

    snapshot_download("OmiKhan/bangla-asr-whisper-medium")
    torchaudio.pipelines.MMS_FA.get_model(with_star=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio")
    .pip_install_from_requirements(str(PROJECT_ROOT / "web" / "requirements.txt"))
    .run_function(download_models)
    # The library uses a src/ layout, so add both packages explicitly for
    # remote Modal containers. add_local_python_source("bangla_asr") cannot
    # discover it from the repository root.
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src", copy=True)
    .add_local_dir(PROJECT_ROOT / "web", remote_path="/root/web", copy=True,
                   ignore=[".venv-gpu", "**/__pycache__"])
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/root/src:/root",
    })
)

volume = modal.Volume.from_name("bangla-asr-web-data", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"
job_store = modal.Dict.from_name("bangla-asr-jobs", create_if_missing=True)

MAX_UPLOAD_MB = 300
MAX_VIDEO_DURATION_S = 15 * 60  # public demo: caps GPU time per upload


@app.cls(image=image, gpu="A10G", volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800, max_containers=1)
class SubtitleWorker:
    """Serverless GPU worker running the SubtitleGenerator."""

    @modal.enter()
    def setup(self):
        from web.backend.pipeline import SubtitleGenerator
        self.generator = SubtitleGenerator(max_duration_s=MAX_VIDEO_DURATION_S)

    @modal.method()
    def process(self, job_id: str, video_relpath: str):
        volume.reload()
        video_full_path = Path(VOLUME_MOUNT_PATH) / video_relpath

        def progress_cb(stage: str, current: int, total: int):
            job_store[job_id] = {
                "status": "processing",
                "stage": stage,
                "progress": {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "percent": int(current / max(1, total) * 100),
                },
                "updated_at": time.time(),
            }

        try:
            result = self.generator.process_video(str(video_full_path), progress_cb=progress_cb)

            srt_relpath = f"srts/{job_id}.srt"
            srt_full_path = Path(VOLUME_MOUNT_PATH) / srt_relpath
            srt_full_path.parent.mkdir(parents=True, exist_ok=True)
            srt_full_path.write_text(result["srt_content"], encoding="utf-8")
            volume.commit()

            job_store[job_id] = {
                "status": "completed",
                "stage": "done",
                "result": {
                    "num_chunks": result["num_chunks"],
                    "total_speech_s": result["total_speech_s"],
                    "cues": result["cues"],
                    "srt_relpath": srt_relpath,
                },
                "completed_at": time.time(),
            }
        except Exception as exc:
            job_store[job_id] = {
                "status": "failed",
                "error": str(exc),
                "failed_at": time.time(),
            }
        finally:
            # the upload is only needed to make the subtitles: never keep users' videos
            video_full_path.unlink(missing_ok=True)
            volume.commit()


fastapi_app = FastAPI(title="Bangla ASR Subtitle Generator")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename or "upload").name
    video_relpath = f"uploads/{job_id}_{safe_name}"
    video_full_path = Path(VOLUME_MOUNT_PATH) / video_relpath
    video_full_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    with open(video_full_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                video_full_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
            f_out.write(chunk)

    volume.commit()

    job_store[job_id] = {
        "status": "pending",
        "stage": "upload",
        "progress": {"stage": "upload", "current": 1, "total": 1, "percent": 100},
        "created_at": time.time(),
    }

    worker = SubtitleWorker()
    worker.process.spawn(job_id, video_relpath)

    return {"job_id": job_id}


@fastapi_app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        data = job_store[job_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **data}


@fastapi_app.get("/jobs/{job_id}/download")
async def download_srt(job_id: str):
    try:
        data = job_store[job_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    if data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not yet complete")

    volume.reload()
    srt_path = Path(VOLUME_MOUNT_PATH) / data["result"]["srt_relpath"]
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="SRT file missing from volume")

    content = srt_path.read_text(encoding="utf-8")
    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
    )


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
@modal.asgi_app()
def web():
    return fastapi_app
