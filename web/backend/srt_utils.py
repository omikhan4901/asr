"""
SRT Subtitle Formatting Utilities for Bangla ASR Subtitles.
"""

from typing import Dict, List


def format_timestamp(seconds: float) -> str:
    """Formats floating-point seconds into standard SRT timestamp HH:MM:SS,mmm."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hrs, remainder = divmod(total_ms, 3600000)
    mins, remainder = divmod(remainder, 60000)
    secs, ms = divmod(remainder, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def chunks_to_srt(subtitled_chunks: List[Dict]) -> str:
    """
    Converts list of chunk dicts:
        [{'start': float, 'end': float, 'text': str}, ...]
    into a valid UTF-8 SubRip (.srt) subtitle string.
    """
    lines = []
    idx = 1
    for chunk in subtitled_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        start_ts = format_timestamp(chunk["start"])
        end_ts = format_timestamp(chunk["end"])
        lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
        idx += 1
    return "\n".join(lines)


def get_srt_preview(srt_text: str, max_entries: int = 5) -> str:
    """Returns the first max_entries subtitle blocks for UI display."""
    blocks = srt_text.strip().split("\n\n")
    return "\n\n".join(blocks[:max_entries])
