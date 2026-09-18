"""
CLI helpers to score existing N-best JSONL pools or generate them from an ASRModel.

Generation is optional and GPU-heavy. The default comparison path is two
precomputed JSONL files (Whisper published pipeline + Wav2Vec2 CTC).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from bangla_asr.common.audio import load_audio
from bangla_asr.models.base import ASRModel


def generate_nbest_records(
    model: ASRModel,
    split_rows: Sequence[Dict[str, Any]],
    n: int,
    *,
    sample_rate: int = 16000,
    model_id: str = "",
) -> List[Dict[str, Any]]:
    """Run ``model`` on a split table and emit comparison-ready JSONL records.

    Each row must include ``utt_id``, ``audio_path``, and ``gt`` (or
    ``ground_truth``). CTC tracks also store ``greedy_text`` so 1-best WER
    does not silently become beam-1 after a language model is attached.
    """
    records: List[Dict[str, Any]] = []
    for row in split_rows:
        audio_path = row["audio_path"]
        try:
            audio, _ = load_audio(audio_path, target_sr=sample_rate)
        except Exception as exc:
            print(f"Skipping {row.get('utt_id', audio_path)}: {exc}")
            continue
        nbest = model.transcribe_nbest(audio, n, sampling_rate=sample_rate)
        greedy = model.transcribe(audio, sampling_rate=sample_rate)
        records.append(
            {
                "utt_id": row.get("utt_id", ""),
                "audio_path": audio_path,
                "ground_truth": row.get("gt", row.get("ground_truth", "")),
                "model_family": model.model_family,
                "model_id": model_id,
                "greedy_text": greedy.text,
                "candidates": [h.to_candidate_dict() for h in nbest],
            }
        )
    return records


def load_split_rows(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a group-aware split CSV into dict rows for ``generate_nbest_records``."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")
