"""
SRT Subtitle Formatting Utilities for Bangla ASR Subtitles.
"""

import unicodedata
from typing import Dict, List

# Common subtitle limits (YouTube, Netflix style): two lines of 42 characters, 1-7 s on screen, readable pace
MAX_LINE_CHARS = 42
MAX_CUE_S = 7.0
MIN_CUE_S = 1.0
MAX_CHARS_PER_S = 20.0
MIN_GAP_S = 0.08
LINGER_S = 0.3        # keep a cue up a little after its last word
BREAK_PAUSE_S = 0.6   # a pause this long always starts a new cue
SENTENCE_END = ("।", "?", "!", ".")
CLAUSE_END = SENTENCE_END + (",", ";", ":")


def visible_len(text: str) -> int:
    """Width on screen in characters: a virama or joiner does not take a place of its own."""
    return sum(1 for ch in text if unicodedata.category(ch) not in ("Mn", "Cf"))


def _two_lines(words: List[str]):
    """(longest line, split index) of the best two-line split: shortest longest line, then the most even."""
    def cost(j):
        a, b = visible_len(" ".join(words[:j])), visible_len(" ".join(words[j:]))
        return max(a, b), abs(a - b)
    j = min(range(1, len(words)), key=cost, default=None)
    return (cost(j)[0], j) if j else (visible_len(" ".join(words)), None)


def _fits(words: List[str]) -> bool:
    return visible_len(" ".join(words)) <= MAX_LINE_CHARS or _two_lines(words)[0] <= MAX_LINE_CHARS


def _split_lines(text: str) -> str:
    """One line if it fits, else two lines split at the best word boundary."""
    words = text.split()
    if visible_len(text) <= MAX_LINE_CHARS:
        return text
    j = _two_lines(words)[1]
    return text if j is None else " ".join(words[:j]) + "\n" + " ".join(words[j:])


def words_to_cues(words: List[Dict], total_s: float) -> List[Dict]:
    """
    Groups timed words [{'text', 'start', 'end'}, ...] into subtitle cues [{'start', 'end', 'text'}, ...].

    A cue ends at a long pause, at a sentence end, or before it would pass two lines or MAX_CUE_S; when it gets
    too long it is cut at the best pause or clause end in its second half, so lines do not break mid-phrase.
    """
    def too_long(group):
        return not _fits([w["text"] for w in group]) or group[-1]["end"] - group[0]["start"] > MAX_CUE_S

    groups, cur = [], []
    for w in words:
        if cur:
            pause = w["start"] - cur[-1]["end"]
            sentence_done = cur[-1]["text"].endswith(SENTENCE_END) and cur[-1]["end"] - cur[0]["start"] >= MIN_CUE_S
            if pause >= BREAK_PAUSE_S or sentence_done:
                groups.append(cur)
                cur = []
            elif too_long(cur + [w]):
                # j = where the next cue starts; j == len(cur) cuts just before w
                cand = range(max(1, len(cur) // 2), len(cur) + 1)
                nxt = lambda j: cur[j] if j < len(cur) else w
                j = max(cand, key=lambda j: (nxt(j)["start"] - cur[j - 1]["end"])
                        + (0.5 if cur[j - 1]["text"].endswith(CLAUSE_END) else 0.0))
                groups.append(cur[:j])
                cur = cur[j:]
        cur.append(w)
    if cur:
        groups.append(cur)

    cues = [{"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(w["text"] for w in g)} for g in groups]
    for i, cue in enumerate(cues):
        limit = (cues[i + 1]["start"] - MIN_GAP_S) if i + 1 < len(cues) else total_s
        wanted = max(cue["end"] + LINGER_S, cue["start"] + MIN_CUE_S,
                     cue["start"] + visible_len(cue["text"]) / MAX_CHARS_PER_S)
        cue["end"] = max(cue["end"], min(wanted, limit, cue["start"] + MAX_CUE_S))
        cue["text"] = _split_lines(cue["text"])
    return cues


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
