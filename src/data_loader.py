"""
Loads the aligned chunk dataset produced by preprocess_chunk_align.py and
splits it into train/eval sets for N-best generation and downstream reranker
training.
"""

import re

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# Chunk audio files are named f"{source_stem}_{chunk_id:03d}.wav" by
# preprocess_chunk_align.py (e.g. "train_001_000.wav" -> source "train_001").
# We recover the source file purely from the filename already present in
# aligned_chunks.csv — no need to touch the chunking script or re-run
# preprocessing, since this info was written into the filename all along.
_CHUNK_SUFFIX_RE = re.compile(r"^(?P<source>.+)_(?P<chunk_id>\d{3})$")


def extract_source_file(audio_path: str) -> str:
    """
    Recovers the original 1-hour source recording's stem (e.g. "train_001")
    from a chunk filename (e.g. "train_001_000.wav"). Falls back to the full
    stem if the expected "_NNN" chunk suffix isn't found, so unexpected
    filenames don't crash the loader — they'll just be treated as their own
    group.
    """
    stem = Path(audio_path).stem
    match = _CHUNK_SUFFIX_RE.match(stem)
    return match.group("source") if match else stem


def load_aligned_dataset(aligned_csv_path: str) -> pd.DataFrame:
    """Loads aligned_chunks.csv: columns [audio_path, gt, align_score]."""
    df = pd.read_csv(aligned_csv_path, encoding="utf-8-sig")
    df = df[df["gt"].notna() & (df["gt"].str.len() > 0)].reset_index(drop=True)
    df["utt_id"] = df["audio_path"].apply(lambda p: Path(p).stem)
    df["source_file"] = df["audio_path"].apply(extract_source_file)
    return df


def train_eval_split(
    df: pd.DataFrame,
    eval_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits at the SOURCE-RECORDING level, not the chunk level. Chunks from
    the same 1-hour source file (same speaker, same recording conditions,
    often overlapping vocabulary from the same script) are kept entirely on
    one side of the split. A naive row-level split would scatter chunks from
    the same recording across both train and eval, letting eval-time
    evaluation benefit from acoustic/textual context the model effectively
    already saw in training — an easy thing for a thesis committee to flag.

    Note: with very few distinct source files (e.g. only train_001,
    train_002 during early testing), GroupShuffleSplit may not hit
    eval_frac exactly — it can only assign whole files to a side. This
    self-corrects once you're running over your full file range.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=eval_frac, random_state=random_state)
    train_idx, eval_idx = next(splitter.split(df, groups=df["source_file"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    eval_df = df.iloc[eval_idx].reset_index(drop=True)

    train_sources = set(train_df["source_file"])
    eval_sources = set(eval_df["source_file"])
    overlap = train_sources & eval_sources
    assert not overlap, f"Source-file leakage across split! Overlapping files: {overlap}"

    return train_df, eval_df


def save_split(train_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_df[["utt_id", "audio_path", "gt"]].to_csv(
        output_path / "train_split.csv", index=False, encoding="utf-8-sig"
    )
    eval_df[["utt_id", "audio_path", "gt"]].to_csv(
        output_path / "eval_split.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"✅ Split saved: {len(train_df)} train chunks "
        f"({train_df['source_file'].nunique()} source files) / "
        f"{len(eval_df)} eval chunks "
        f"({eval_df['source_file'].nunique()} source files) → {output_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_csv", type=str, default="data/processed/chunks/aligned_chunks.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/splits")
    parser.add_argument("--eval_frac", type=float, default=0.2)
    args = parser.parse_args()

    df = load_aligned_dataset(args.aligned_csv)
    train_df, eval_df = train_eval_split(df, eval_frac=args.eval_frac)
    save_split(train_df, eval_df, args.output_dir)