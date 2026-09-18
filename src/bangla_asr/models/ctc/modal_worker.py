"""
Distributed Wav2Vec2 CTC N-best generation on Modal.

Reuses the existing bangla-asr-data volume and shard/merge pattern from the
Whisper worker. The container image adds ``pyctcdecode`` so N-best cannot
silently degrade to greedy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import modal

from bangla_asr.config import Wav2Vec2Config

PROJECT_ROOT = Path(__file__).resolve().parents[4]

app = modal.App("bangla-asr-wav2vec2-nbest")

SHARD_TIMEOUT_S = 6 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "librosa",
        "soundfile",
        "pandas",
        "numpy",
        "jiwer",
        "pyctcdecode",
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
class Wav2Vec2ModalWorker:
    """Parallel worker running Wav2Vec2 CTC beam search on one shard."""

    @modal.enter()
    def initialize(self):
        from bangla_asr.config import Wav2Vec2Config
        from bangla_asr.models.ctc.wav2vec2_model import Wav2Vec2Model

        self.cfg = Wav2Vec2Config()
        self.model = Wav2Vec2Model(self.cfg)

    @modal.method()
    def process_shard(self, shard_records: List[Dict], shard_id: int, output_jsonl_name: str) -> Dict:
        import torch
        from bangla_asr.common.audio import load_audio

        torch.manual_seed(BASE_SEED + shard_id)
        shard_out_path = os.path.join(VOLUME_MOUNT_PATH, f"{output_jsonl_name}.shard{shard_id:02d}.jsonl")

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
        batch_size = self.cfg.batch_size
        n_written = 0
        n = self.cfg.beam_width

        with open(shard_out_path, "a", encoding="utf-8") as f_out:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                audio_arrays = []
                valid_records = []

                for rec in batch:
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
                    except Exception as exc:
                        print(f"[Shard {shard_id}] Failed loading audio {audio_path}: {exc}")

                if not audio_arrays:
                    continue

                try:
                    greedy_hyps = self.model.transcribe_nbest_batch(audio_arrays, n=1)
                    nbest_hyps = self.model.transcribe_nbest_batch(audio_arrays, n=n)
                except Exception as exc:
                    print(f"[Shard {shard_id}] CTC decode failed: {exc}")
                    torch.cuda.empty_cache()
                    continue

                for rec, greedy, cands in zip(valid_records, greedy_hyps, nbest_hyps):
                    record = {
                        "utt_id": rec["utt_id"],
                        "audio_path": rec["audio_path"],
                        "ground_truth": rec.get("gt", rec.get("ground_truth", "")),
                        "model_family": self.model.model_family,
                        "model_id": self.cfg.model_id,
                        "greedy_text": greedy[0].text if greedy else "",
                        "candidates": [h.to_candidate_dict() for h in cands],
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

                f_out.flush()
                if n_written % 20 < batch_size:
                    volume.commit()

        volume.commit()
        return {"shard_id": shard_id, "n_written": n_written, "shard_out_path": shard_out_path}
