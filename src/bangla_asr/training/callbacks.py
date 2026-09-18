"""
Training callbacks: JSON/CSV logging, checkpointing, held-out eval, early stopping.

Weights & Biases is intentionally not supported.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonCsvLogger:
    """Append-only step logs written as JSONL and a mirrored CSV."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train_log.jsonl"
        self.csv_path = self.log_dir / "train_log.csv"
        self._csv_fields: Optional[List[str]] = None

    def log(self, row: Dict[str, Any]) -> None:
        """Write one step/eval record to both log files."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            return
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writerow(row)


@dataclass
class EarlyStopping:
    """Stop when the monitored eval metric does not improve for ``patience`` evals."""

    patience: int
    mode: str = "max"
    best: Optional[float] = None
    bad_evals: int = 0
    should_stop: bool = False
    history: List[float] = field(default_factory=list)

    def update(self, value: float) -> bool:
        """Record ``value`` and return True when training should stop."""
        self.history.append(value)
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best
        else:
            improved = value < self.best
        if improved:
            self.best = value
            self.bad_evals = 0
        else:
            self.bad_evals += 1
            if self.bad_evals >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint(payload: Dict[str, Any], checkpoint_dir: str, name: str) -> Path:
    """Write a JSON sidecar for a checkpoint. Model weights are saved by the trainer."""
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
