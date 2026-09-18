"""
Dataset loader registry.

Bangla long-form manifests are the primary track and reuse
``dataset.manifest.load_aligned_manifest``. English clinical corpora
(MedDialog-EN-ASR, PriMock-style) are an optional *parallel* track: this
module exposes named hooks so they can be plugged in later with their own
loader and their own SWER lexicon. Those loaders are not implemented here
and must not change Bangla config defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bangla_asr.dataset.manifest import load_aligned_manifest


@dataclass
class UtteranceRecord:
    """Language-tagged utterance used by generation and training entry points."""

    utt_id: str
    audio_path: str
    ground_truth: str
    language: str
    source: str
    extras: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader(ABC):
    """Minimal loader interface. New corpora add a subclass; they do not fork configs."""

    name: str
    language: str

    @abstractmethod
    def load(self, path: str, limit: Optional[int] = None) -> List[UtteranceRecord]:
        """Read a split/manifest from ``path`` into utterance records."""


class BanglaManifestLoader(DatasetLoader):
    """Primary track: aligned Bangla chunk CSV from the existing preprocess pipeline."""

    name = "bangla_manifest"
    language = "bn"

    def load(self, path: str, limit: Optional[int] = None) -> List[UtteranceRecord]:
        df = load_aligned_manifest(path)
        if limit is not None:
            df = df.head(limit)
        records: List[UtteranceRecord] = []
        for row in df.to_dict(orient="records"):
            records.append(
                UtteranceRecord(
                    utt_id=str(row.get("utt_id", "")),
                    audio_path=str(row.get("audio_path", "")),
                    ground_truth=str(row.get("gt", "")),
                    language=self.language,
                    source=str(row.get("source_file", self.name)),
                    extras={"align_score": row.get("align_score")},
                )
            )
        return records


class EnglishClinicalLoader(DatasetLoader):
    """Placeholder for MedDialog-EN-ASR / PriMock. Not implemented on purpose."""

    language = "en"

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self, path: str, limit: Optional[int] = None) -> List[UtteranceRecord]:
        raise NotImplementedError(
            f"Loader {self.name!r} is an optional English clinical parallel track. "
            "It is not implemented and must not be used for the Bangla baseline "
            "(34.85% → 32.29%). Add a dedicated parser and an English SWER lexicon "
            f"before reading {path}."
        )


def get_loader(name: str) -> DatasetLoader:
    """Return a loader by registry name. Unknown names raise ``ValueError``."""
    key = (name or "").strip().lower()
    if key in {"bangla_manifest", "bangla", "bn"}:
        return BanglaManifestLoader()
    if key in {"meddialog_en_asr", "primock", "english_clinical"}:
        return EnglishClinicalLoader(key)
    raise ValueError(
        f"Unknown dataset loader {name!r}. Bangla baseline: 'bangla_manifest'. "
        "English clinical hooks (unimplemented): 'meddialog_en_asr', 'primock'."
    )
