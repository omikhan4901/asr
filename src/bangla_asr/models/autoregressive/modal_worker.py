"""
Distributed N-best candidate generation on Modal across parallel GPU containers.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

import modal

from bangla_asr.config import WhisperConfig, AudioConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]

app = modal.App("bangla-asr-nbest-generator")

SHARD_TIMEOUT_S = 6 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "accelerate", "librosa",
        "soundfile", "pandas", "numpy", "rapidfuzz", "jiwer",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src", copy=True)
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/root/src",
    })
)

volume = modal.Volume.from_name("bangla-asr-data", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"
BASE_SEED = 1234


@app.cls(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=SHARD_TIMEOUT_S,
    max_containers=10,
)
class DistributedModalWorker:
    """Parallel worker container running Whisper on an allocated shard of audio chunks."""

    @modal.enter()
    def initialize(self):
        from bangla_asr.config import WhisperConfig
        from bangla_asr.models.autoregressive.whisper_model import WhisperASRModel
        from bangla_asr.models.autoregressive.generator import BatchedNBestGenerator

        self.whisper_cfg = WhisperConfig()
        self.model_wrapper = WhisperASRModel(self.whisper_cfg)
        self.generator = BatchedNBestGenerator(self.model_wrapper)

    @modal.method()
    def process_shard(self, shard_records: List[Dict], shard_id: int, output_jsonl_name: str) -> Dict:
        import torch
        from bangla_asr.common.audio import load_audio

        torch.manual_seed(BASE_SEED + shard_id)
        shard_out_path = os.path.join(VOLUME_MOUNT_PATH, f"{output_jsonl_name}.shard{shard_id:02d}.jsonl")

        # Resume support: detect already processed utterance IDs
        completed_ids = set()
        if os.path.exists(shard_out_path):
            with open(shard_out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            completed_ids.add(json.loads(line)["utt_id"])
                        except Exception:
                            pass

        remaining = [r for r in shard_records if r["utt_id"] not in completed_ids]
        batch_size = self.whisper_cfg.batch_size
        n_written = 0

        with open(shard_out_path, "a", encoding="utf-8") as f_out:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                audio_arrays = []
                valid_records = []

                for rec in batch:
                    # Resolve audio path on Modal volume
                    rel_p = rec["audio_path"].replace("\\", "/").lstrip("/")
                    possible_paths = [
                        os.path.join(VOLUME_MOUNT_PATH, rel_p),
                        os.path.join(VOLUME_MOUNT_PATH, Path(rel_p).name),
                        rec["audio_path"],
                    ]
                    audio_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

                    try:
                        arr, _ = load_audio(audio_path, target_sr=16000)
                        audio_arrays.append(arr)
                        valid_records.append(rec)
                    except Exception as e:
                        print(f"[Shard {shard_id}] Failed loading audio {audio_path}: {e}")

                if not audio_arrays:
                    continue

                try:
                    candidates_batch = self.generator.generate_batch(audio_arrays)
                except Exception as e:
                    print(f"[Shard {shard_id}] Generation failed on batch starting at {valid_records[0]['utt_id']}: {e}")
                    torch.cuda.empty_cache()
                    continue

                for rec, cands in zip(valid_records, candidates_batch):
                    record = {
                        "utt_id": rec["utt_id"],
                        "audio_path": rec["audio_path"],
                        "ground_truth": rec.get("gt", rec.get("ground_truth", "")),
                        "candidates": cands,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

                f_out.flush()
                if n_written % 20 < batch_size:
                    volume.commit()

        volume.commit()
        return {"shard_id": shard_id, "n_written": n_written, "shard_out_path": shard_out_path}
