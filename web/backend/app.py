"""
Deployable Modal app for the Bangla subtitle generator.

Architecture:
  - `web` (CPU, always-on-ish FastAPI app via asgi_app): handles file
    upload, saves it to a Volume, writes an initial job entry, and
    .spawn()'s the GPU worker to process it in the background — returns
    immediately so the frontend can start polling. Also serves the
    status-check and download endpoints.
  - `SubtitleWorker` (GPU class, A10G): loads the Whisper model once per
    container (@modal.enter), then processes one video per job, updating
    the shared job-status Dict at each pipeline stage so the frontend's
    polling loop reflects real progress, not a fake progress bar.

Deploy: modal deploy app.py
Then set window.ASR_API_BASE in the frontend to the printed web URL.
"""

import io
import json
import time
import traceback
import uuid
from pathlib import Path

import modal

app = modal.App("bangla-subtitle-generator")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "transformers", "accelerate", "librosa", "soundfile",
        "jiwer", "numpy", "fastapi[standard]", "python-multipart",
        "silero-vad",
    )
    .add_local_file("pipeline.py", "/root/pipeline.py")
    .add_local_file("srt_utils.py", "/root/srt_utils.py")
)

volume = modal.Volume.from_name("subtitle-web-data", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"

# job_id -> {"status", "stage", "progress": {"current","total"}, "error", "result"}
job_store = modal.Dict.from_name("subtitle-web-jobs", create_if_missing=True)

WEB_ORIGINS = ["*"]  # tighten to your actual frontend domain before real deployment


def _update_job(job_id: str, **fields):
    current = job_store.get(job_id, {})
    current.update(fields)
    job_store[job_id] = current


@app.cls(image=image, gpu="A10G", volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800, max_containers=5)
class SubtitleWorker:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        from pipeline import SubtitleGenerator
        self.generator = SubtitleGenerator()

    @modal.method()
    def process(self, job_id: str, video_relpath: str):
        # Container may be warm/reused (max_containers=5) — its volume view
        # was mounted whenever IT started, not when this video was just
        # uploaded by a different (web) container. Reload so we see the
        # video the web endpoint actually just wrote. Same fix already
        # applied to the download endpoint below; this was the one place
        # it was missing.
        volume.reload()
        video_path = str(Path(VOLUME_MOUNT_PATH) / video_relpath)

        def progress_cb(stage, current=None, total=None):
            fields = {"status": "processing", "stage": stage}
            if current is not None and total is not None:
                fields["progress"] = {"current": current, "total": total}
            _update_job(job_id, **fields)

        try:
            _update_job(job_id, status="processing", stage="audio")
            result = self.generator.process_video(video_path, progress_cb=progress_cb)

            srt_relpath = f"outputs/{job_id}.srt"
            srt_full_path = Path(VOLUME_MOUNT_PATH) / srt_relpath
            srt_full_path.parent.mkdir(parents=True, exist_ok=True)
            srt_full_path.write_text(result["srt_text"], encoding="utf-8")
            volume.commit()

            _update_job(
                job_id,
                status="done",
                stage="srt",
                result={
                    "n_segments": result["n_segments"],
                    "duration_seconds": result["duration_seconds"],
                    "srt_preview": result["srt_preview"],
                    "srt_relpath": srt_relpath,
                },
            )
        except Exception as e:
            traceback.print_exc()  # otherwise `modal app logs` shows only
                                     # the bare message, no idea where it failed
            _update_job(job_id, status="error", error=str(e))


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware, allow_origins=WEB_ORIGINS,
        allow_methods=["*"], allow_headers=["*"],
    )

    MAX_UPLOAD_MB = 300
    MAX_DURATION_SECONDS = 600  # 10 min — protects both UX (worker
                                  # timeout=1800 could plausibly be exceeded
                                  # by longer video at num_beams=12) and
                                  # remaining compute budget

    @fastapi_app.post("/jobs")
    async def create_job(file: UploadFile = File(...)):
        job_id = uuid.uuid4().hex[:12]

        # file.filename is attacker-controlled — Path(...).name strips any
        # directory components (e.g. "../../evil.wav") so it can't escape
        # uploads/ when written below.
        safe_name = Path(file.filename or "upload").name
        video_relpath = f"uploads/{job_id}_{safe_name}"
        video_full_path = Path(VOLUME_MOUNT_PATH) / video_relpath
        video_full_path.parent.mkdir(parents=True, exist_ok=True)

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")

        video_full_path.write_bytes(contents)
        volume.commit()

        # Check duration BEFORE spawning a GPU worker — no point burning
        # A10G time on a job we're about to reject anyway.
        import subprocess
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrapped_values=1:nokey=1", str(video_full_path)],
            capture_output=True, text=True,
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = None  # couldn't probe — let the worker's own error handling catch it

        if duration is not None and duration > MAX_DURATION_SECONDS:
            video_full_path.unlink(missing_ok=True)
            volume.commit()
            raise HTTPException(
                status_code=413,
                detail=f"Video is {duration/60:.1f} min — this demo currently caps at {MAX_DURATION_SECONDS//60} min.",
            )

        job_store[job_id] = {"status": "queued", "stage": "upload", "result": None, "error": None}

        SubtitleWorker().process.spawn(job_id, video_relpath)

        return {"job_id": job_id}

    @fastapi_app.get("/jobs/{job_id}")
    async def get_status(job_id: str):
        if job_id not in job_store:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_store[job_id]

    @fastapi_app.get("/jobs/{job_id}/download")
    async def download(job_id: str):
        job = job_store.get(job_id)
        if not job or job.get("status") != "done":
            raise HTTPException(status_code=404, detail="Result not ready")

        volume.reload()
        srt_path = Path(VOLUME_MOUNT_PATH) / job["result"]["srt_relpath"]
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail="File missing on volume")

        return Response(
            content=srt_path.read_text(encoding="utf-8"),
            media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
        )

    return fastapi_app