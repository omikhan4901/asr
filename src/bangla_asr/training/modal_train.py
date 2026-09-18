"""
Modal GPU job for GRPO correction training.

Reuses the bangla-asr-data volume and debian-slim + pip image pattern from
the Whisper N-best worker. No new deployment abstraction.
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[3]

app = modal.App("bangla-asr-grpo")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "jiwer",
        "pyyaml",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src", copy=True)
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/root/src",
    })
)

volume = modal.Volume.from_name("bangla-asr-data", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=6 * 60 * 60,
)
def train_grpo_job(
    train_jsonl: str,
    eval_jsonl: str = "",
    max_steps: int = 100,
    group_size: int = 4,
    learning_rate: float = 1e-6,
) -> dict:
    """Run GRPO on Modal and write checkpoints/logs back to the shared volume."""
    from bangla_asr.config import GRPOConfig
    from bangla_asr.training.train_grpo import run_training_from_jsonl

    cfg = GRPOConfig(
        max_steps=max_steps,
        group_size=group_size,
        learning_rate=learning_rate,
        checkpoint_dir=f"{VOLUME_MOUNT_PATH}/checkpoints/grpo",
        log_dir=f"{VOLUME_MOUNT_PATH}/logs/grpo",
    )
    eval_path = eval_jsonl or None
    result = run_training_from_jsonl(train_jsonl, eval_path, config=cfg)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    train_jsonl: str = "/data/train_correction.jsonl",
    eval_jsonl: str = "",
    max_steps: int = 100,
):
    print(train_grpo_job.remote(train_jsonl, eval_jsonl, max_steps))
