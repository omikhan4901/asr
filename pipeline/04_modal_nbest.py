#!/usr/bin/env python3
"""
Step 4: Distributed N-Best Candidate Generation via Modal (Parallel A10G Sharding).
"""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd
import modal

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bangla_asr.models.autoregressive.modal_worker import app, DistributedModalWorker, volume, VOLUME_MOUNT_PATH


@app.local_entrypoint()
def main(
    split_csv_path: str = "train_split.csv",
    output_jsonl: str = "train_nbest_candidates.jsonl",
    num_shards: int = 10,
    sample_n: int = None,
):
    csv_full_path = os.path.join(VOLUME_MOUNT_PATH, split_csv_path)
    print(f"Reading dataset split from volume: {csv_full_path}")
    df = pd.read_csv(csv_full_path, encoding="utf-8-sig")
    records = df.to_dict(orient="records")

    if sample_n:
        records = records[:sample_n]
        print(f"Sample limit enabled: processing {sample_n} records.")

    # Evenly partition records across shards
    shards = [records[i::num_shards] for i in range(num_shards)]
    shards = [s for s in shards if s]
    print(f"Launching {len(shards)} parallel workers on Modal...")

    worker = DistributedModalWorker()
    t0 = time.time()
    results = list(worker.process_shard.starmap([(shard, i, output_jsonl) for i, shard in enumerate(shards)]))
    elapsed = time.time() - t0

    print(f"Parallel shard execution completed in {elapsed:.1f}s.")
    volume.reload()

    # Deterministic ordered merge back to main JSONL
    merged_path = os.path.join(VOLUME_MOUNT_PATH, output_jsonl)
    total_written = 0
    with open(merged_path, "w", encoding="utf-8") as f_out:
        for r in results:
            shard_p = r["shard_out_path"]
            if os.path.exists(shard_p):
                with open(shard_p, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if line.strip():
                            f_out.write(line)
                            total_written += 1

    volume.commit()
    print(f"Merged {total_written} total candidate records into {merged_path}")


if __name__ == "__main__":
    pass
