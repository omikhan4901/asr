"""
Manifest ingestion, validation, and serialization utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

from bangla_asr.dataset.splitter import extract_source_id


def load_aligned_manifest(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads aligned_chunks.csv, cleans empty ground truth, and populates source IDs.
    Required columns: ['audio_path', 'gt', 'align_score'].
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Clean nulls or empty ground truth
    df = df[df["gt"].notna() & (df["gt"].astype(str).str.strip().str.len() > 0)].reset_index(drop=True)
    df["utt_id"] = df["audio_path"].apply(lambda p: Path(p).stem)
    df["source_file"] = df["audio_path"].apply(extract_source_id)
    return df


def load_nbest_jsonl(jsonl_path: Union[str, Path]) -> List[Dict]:
    """Reads JSONL N-best records containing utt_id, audio_path, ground_truth, and candidates."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_nbest_jsonl(records: List[Dict], jsonl_path: Union[str, Path]) -> None:
    """Saves candidate records to a JSONL file."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
