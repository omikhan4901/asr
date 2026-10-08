"""
Run records: what produced a result file, so every number can be traced.

    with RunRecord("results/x/run_info.json", inputs=[manifest]) as rec:
        ...  # the work
        rec.outputs = [pred_path, metrics_path]

writes (or extends, for resumed runs) a JSON file with: git commit and dirty
flag, command line, start/end time and runtime, Python/torch/transformers
versions, GPU name, and SHA-256 of every input and output file. Each run
(also each resume) appends one entry to "runs"; the newest is last.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Union

from bangla_asr.common.safe_io import atomic_write_text

PathLike = Union[str, Path]


def sha256(path: PathLike) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "numpy", "pandas"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


class RunRecord:
    def __init__(self, record_path: PathLike, inputs: Iterable[PathLike] = ()):
        self.path = Path(record_path)
        self.inputs: List[PathLike] = list(inputs)
        self.outputs: List[PathLike] = []
        self.extra: dict = {}

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        entry = {
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runtime_s": round(time.time() - self.t0, 1),
            "status": "ok" if exc_type is None else f"failed: {exc_type.__name__}",
            "versions": _versions(),
            "inputs": {str(p): sha256(p) for p in self.inputs},
            "outputs": {str(p): sha256(p) for p in self.outputs},
            **self.extra,
        }
        record = {"runs": []}
        if self.path.exists():
            try:
                record = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        record["runs"].append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(record, ensure_ascii=False, indent=2))
        return False  # never swallow the exception
