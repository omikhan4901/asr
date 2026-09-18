"""
Load a clinical-severity lexicon for SWER.

v1 matching is exact after ``normalize_bangla_text``. This table is *not*
Roy 2021 Semantic WER: it applies explicit clinical multipliers, not
embedding/semantic similarity.

The shipped Bangla table is a PLACEHOLDER. English clinical lexicons
(MedDialog / PriMock) are a parallel track and are not loaded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from bangla_asr.common.text import normalize_bangla_text

PACKAGE_PLACEHOLDER_YAML = Path(__file__).resolve().parent / "lexicons" / "placeholder_bangla.yaml"

# Embedded copy so tests and default scoring run without PyYAML.
PLACEHOLDER_LEXICON: Dict[str, Any] = {
    "version": "placeholder-v1",
    "language": "bn",
    "placeholder": True,
    "critical_tokens": ["ইনসুলিন", "ডোজ", "না", "উচ্চরক্তচাপ", "নিম্নরক্তচাপ"],
    "critical_phrases": ["উচ্চ রক্তচাপ", "নিম্ন রক্তচাপ"],
    "critical_substitution_pairs": [
        ["উচ্চরক্তচাপ", "নিম্নরক্তচাপ"],
        ["উচ্চ", "নিম্ন"],
        ["না", "হ্যাঁ"],
    ],
    "filler_tokens": ["মানে", "আসলে", "হুম", "যে"],
    "garbled_tokens": ["ইনসলন", "ডোজ্সস"],
    "known_words": [],
}


def _norm_token(token: str) -> str:
    return normalize_bangla_text(token)


def _norm_tokens(values: Iterable[str]) -> Set[str]:
    return {_norm_token(v) for v in values if _norm_token(v)}


@dataclass
class SeverityLexicon:
    """Normalized lookup tables for SWER edit classification."""

    source: str
    language: str
    placeholder: bool
    critical_tokens: Set[str] = field(default_factory=set)
    critical_phrases: List[Tuple[str, ...]] = field(default_factory=list)
    critical_substitution_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    filler_tokens: Set[str] = field(default_factory=set)
    garbled_tokens: Set[str] = field(default_factory=set)
    known_words: Set[str] = field(default_factory=set)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any], source: str) -> "SeverityLexicon":
        """Build a lexicon from a YAML/JSON mapping, normalizing every token."""
        pairs: Set[Tuple[str, str]] = set()
        for pair in data.get("critical_substitution_pairs") or []:
            if len(pair) != 2:
                raise ValueError(f"critical_substitution_pairs entries must be length 2, got {pair!r}")
            left, right = _norm_token(pair[0]), _norm_token(pair[1])
            if left and right:
                pairs.add((left, right))
                pairs.add((right, left))

        phrases: List[Tuple[str, ...]] = []
        for phrase in data.get("critical_phrases") or []:
            toks = tuple(_norm_token(phrase).split())
            if toks:
                phrases.append(toks)

        return cls(
            source=source,
            language=str(data.get("language", "bn")),
            placeholder=bool(data.get("placeholder", False)),
            critical_tokens=_norm_tokens(data.get("critical_tokens") or []),
            critical_phrases=phrases,
            critical_substitution_pairs=pairs,
            filler_tokens=_norm_tokens(data.get("filler_tokens") or []),
            garbled_tokens=_norm_tokens(data.get("garbled_tokens") or []),
            known_words=_norm_tokens(data.get("known_words") or []),
        )

    def mark_critical_reference_tokens(self, ref_tokens: Sequence[str]) -> List[bool]:
        """True where a reference unigram or listed phrase is clinically critical."""
        flags = [tok in self.critical_tokens for tok in ref_tokens]
        n = len(ref_tokens)
        for phrase in self.critical_phrases:
            width = len(phrase)
            if width == 0 or width > n:
                continue
            for start in range(0, n - width + 1):
                if tuple(ref_tokens[start:start + width]) == phrase:
                    for idx in range(start, start + width):
                        flags[idx] = True
        return flags

    def is_critical_substitution(self, ref_token: str, hyp_token: str) -> bool:
        return (ref_token, hyp_token) in self.critical_substitution_pairs

    def is_filler(self, token: str) -> bool:
        return token in self.filler_tokens

    def is_garbled(self, token: str, treat_oov_as_garbled: bool = False) -> bool:
        """Return whether ``token`` is an unintelligible / collapsed form.

        OOV-as-garbled is off by default because the placeholder ``known_words``
        list is empty; turning it on would mark almost all Bangla as garbled.
        """
        if token in self.garbled_tokens:
            return True
        if not treat_oov_as_garbled:
            return False
        if not self.known_words:
            return False
        return token not in self.known_words and token not in self.filler_tokens


def _read_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                f"YAML lexicon {path} requires PyYAML. Install with: pip install pyyaml "
                "or pass a .json lexicon."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Lexicon file must contain a mapping, got {type(data)}")
    return data


def load_severity_lexicon(path: Optional[str] = None) -> SeverityLexicon:
    """Load a lexicon file, or the embedded Bangla PLACEHOLDER if ``path`` is None.

    English clinical lexicons must be a separate file and a separate loader.
    This function will not silently switch language.
    """
    if path is None:
        yaml_path = PACKAGE_PLACEHOLDER_YAML
        if yaml_path.exists():
            try:
                return SeverityLexicon.from_mapping(_read_mapping(yaml_path), source=str(yaml_path))
            except ImportError:
                pass
        return SeverityLexicon.from_mapping(
            PLACEHOLDER_LEXICON,
            source="PLACEHOLDER:embedded",
        )

    lexicon_path = Path(path)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Severity lexicon not found: {lexicon_path}")
    return SeverityLexicon.from_mapping(_read_mapping(lexicon_path), source=str(lexicon_path))
