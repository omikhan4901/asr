#!/usr/bin/env python3
"""
Step 4b: Distributed Wav2Vec2 CTC N-best on Modal.

Sister script of pipeline/04_modal_nbest.py. Does not change the Whisper job.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.models.ctc.modal_worker import (
    VOLUME_MOUNT_PATH,
    Wav2Vec2ModalWorker,
    app,
    volume,
)


@app.local_entrypoint()
def main(
    split_csv_path: str = "eval_split.csv",
    output_jsonl: str = "eval_wav2vec2_nbest.jsonl",
    num_shards: int = 10,
    sample_n: int = None,
):
    csv_full_path = os.path.join(VOLUME_MOUNT_PATH, split_csv_path)
    df = pd.read_csv(csv_full_path, encoding="utf-8-sig")
    records = df.to_dict(orient="records")
    if sample_n:
        records = records[:sample_n]
    shards = [s for s in (records[i::num_shards] for i in range(num_shards)) if s]
    worker = Wav2Vec2ModalWorker()
    t0 = time.time()
    results = list(worker.process_shard.starmap([(shard, i, output_jsonl) for i, shard in enumerate(shards)]))
    print(f"Finished in {time.time() - t0:.1f}s")
    volume.reload()
    merged_path = os.path.join(VOLUME_MOUNT_PATH, output_jsonl)
    total = 0
    with open(merged_path, "w", encoding="utf-8") as f_out:
        for r in results:
            if os.path.exists(r["shard_out_path"]):
                with open(r["shard_out_path"], "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if line.strip():
                            f_out.write(line)
                            total += 1
    volume.commit()
    print(f"Merged {total} records into {merged_path}")
