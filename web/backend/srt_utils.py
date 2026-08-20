"""
Converts a list of rescored chunks into a valid .srt file.

Each chunk is expected to have: {"start": float_seconds, "end": float_seconds,
"text": str}. This is chunk-level granularity (up to 25s per entry) — see
the paper's stated limitation; sentence-level splitting via word-level
timestamps is noted as future work, not implemented here.
"""


def _format_timestamp(seconds: float) -> str:
    """
    SRT format: HH:MM:SS,mmm

    Computed from total milliseconds up front so second/minute/hour carries
    (e.g. rounding 59.9997s up to the next whole second) resolve correctly
    in one place, instead of being patched as a special case.
    """
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def chunks_to_srt(chunks: list[dict]) -> str:
    """
    chunks: list of {"start": float, "end": float, "text": str}, assumed
    already in chronological order (matches Phase 1's monotonic chunking).
    """
    lines = []
    entry_num = 0  # separate counter — only increments for entries actually
                    # written, so skipped (empty-text) chunks don't leave
                    # gaps in the SRT's sequence numbers (e.g. 1,2,4,5,...)
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        entry_num += 1
        start = _format_timestamp(chunk["start"])
        end = _format_timestamp(chunk["end"])
        lines.append(str(entry_num))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line between entries
    return "\n".join(lines)


def srt_preview(srt_text: str, max_entries: int = 5) -> str:
    """First few entries only, for the frontend preview panel."""
    blocks = srt_text.strip().split("\n\n")
    return "\n\n".join(blocks[:max_entries])