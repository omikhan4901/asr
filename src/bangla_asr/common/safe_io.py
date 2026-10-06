"""
Crash-safe file helpers, so long jobs survive power cuts.

- ``atomic_write_text``: write to a temp file, then rename over the target.
  A crash leaves either the old file or the new one, never half a file.
- ``repair_jsonl``: drop a half-written last line left by a crash, so the
  file can be appended to again.
- ``read_jsonl``: read records, skipping any malformed line.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def repair_jsonl(path: PathLike) -> None:
    """Truncate a trailing partial line (no final newline) left by an interrupted write."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        cut = data.rfind(b"\n") + 1  # 0 if there is no complete line at all
        f.seek(cut)
        f.truncate()


def read_jsonl(path: PathLike) -> List[Dict]:
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
