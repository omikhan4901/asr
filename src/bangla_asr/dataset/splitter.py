"""
Group-aware train/evaluation dataset splitting to prevent recording-level data leakage.
"""

import re
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

# Matches source recording stem and any arbitrary numeric chunk suffix (e.g., train_001_042 or train_001_1250)
_CHUNK_STEM_REGEX = re.compile(r"^(?P<source>.+?)_(?P<chunk_id>\d+)$")


def extract_source_id(audio_path_or_stem: str) -> str:
    """Extracts the parent recording identifier from a chunk path or stem."""
    stem = Path(audio_path_or_stem).stem
    match = _CHUNK_STEM_REGEX.match(stem)
    return match.group("source") if match else stem


def group_aware_split(
    df: pd.DataFrame,
    eval_fraction: float = 0.2,
    random_seed: int = 42,
    group_column: str = "source_file",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataframe by groups (source files) so no source recording spans both splits.
    """
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in dataframe")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be strictly between 0 and 1")

    unique_groups = df[group_column].unique()
    if len(unique_groups) < 2:
        raise ValueError("At least two source groups are required for a leakage-free split")
    np.random.seed(random_seed)
    shuffled_groups = np.random.permutation(unique_groups)

    n_eval_groups = int(round(len(shuffled_groups) * eval_fraction))
    n_eval_groups = min(len(shuffled_groups) - 1, max(1, n_eval_groups))
    eval_groups = set(shuffled_groups[:n_eval_groups])
    train_groups = set(shuffled_groups[n_eval_groups:])

    # Strict leakage validation
    overlap = train_groups.intersection(eval_groups)
    assert not overlap, f"Fatal split leakage detected: {overlap}"

    train_df = df[df[group_column].isin(train_groups)].reset_index(drop=True)
    eval_df = df[df[group_column].isin(eval_groups)].reset_index(drop=True)

    return train_df, eval_df
