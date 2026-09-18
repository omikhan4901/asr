"""
Swappable detectors for the SWER "garbled / unintelligible" bin.

Why this exists
---------------
CTC models (Wav2Vec2) typically destroy a word into phonetic collapse —
not a fluent near-synonym. The garbled bin must catch that failure mode.
An explicit ``garbled_tokens`` list will never list the real errors.

Strategies (tradeoffs)
----------------------
``explicit``
    Only tokens listed in the severity lexicon. Zero false positives, near-total
    false negatives on real CTC output. Kept for debugging.

``character``
    Bangla orthography checks (repeated runs, illegal matra/virama sequences).
    Cheap, no wordlist. High precision on collapse artifacts; misses
    plausible-looking but meaningless tokens (false negatives).

``wordlist_oov``
    Any token absent from the wordlist is garbled. Cheap given a small list.
    High false positives until the wordlist is large (names, inflections, drugs).

``edit_distance``
    OOV *and* character-Levenshtein distance to the nearest wordlist entry
    exceeds a threshold. Protects minor phonetic near-misses. Needs a wordlist;
    cost is O(|wordlist| * |token|^2) per token (fine for a few thousand words).

``composite`` (default)
    explicit-list OR character-implausible OR (OOV and far from the wordlist).
    Recommendation: this is the default because it catches CTC collapse
    without treating every unseen real word as garbled, and it stays
    swappable via ``SWERConfig.garbled_detector``.

The shipped wordlist is a STARTING POINT (high-frequency / test / placeholder
clinical terms), not a dictionary. Replace it before any paper claim.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

from bangla_asr.common.text import normalize_bangla_text

STARTER_WORDLIST_PATH = Path(__file__).resolve().parent / "lexicons" / "starter_bangla_wordlist.txt"

# Bengali dependent vowel signs (matras) and virama (hasanta).
_MATRA = set("ািীুূৃেৈোৌৗ")
_VIRAMA = "্"
_REPEAT_RUN = re.compile(r"(.)\1\1")


def _norm(token: str) -> str:
    return normalize_bangla_text(token)


def load_bangla_wordlist(path: Optional[str] = None) -> Set[str]:
    """Load a one-word-per-line Bangla list. ``#`` starts a comment.

    Defaults to the shipped starter list. Missing files yield an empty set
    so character/explicit detectors still run.
    """
    wordlist_path = Path(path) if path else STARTER_WORDLIST_PATH
    if not wordlist_path.exists():
        return set()
    words: Set[str] = set()
    for raw in wordlist_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        token = _norm(line)
        if token:
            words.add(token)
    return words


def character_levenshtein(left: str, right: str) -> int:
    """Character-level Wagner–Fischer distance for near-miss gating."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        curr = [i] + [0] * len(right)
        for j, rch in enumerate(right, start=1):
            cost = 0 if lch == rch else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def min_edit_distance(token: str, wordlist: Sequence[str], length_slack: int) -> Optional[int]:
    """Nearest-neighbour edit distance, skipping words of very different length."""
    if not wordlist:
        return None
    n = len(token)
    best: Optional[int] = None
    for word in wordlist:
        if abs(len(word) - n) > length_slack:
            continue
        dist = character_levenshtein(token, word)
        if best is None or dist < best:
            best = dist
            if best == 0:
                return 0
    return best


def is_character_implausible(token: str) -> bool:
    """Cheap Bangla orthography pre-filter for collapsed / illegal tokens.

    True when the token is unlikely to be a real written word:
    three-plus identical character run, leading matra/virama, stacked matras,
    stacked viramas, or a trailing virama (incomplete conjunct).
    """
    if not token:
        return False
    if _REPEAT_RUN.search(token):
        return True
    if token[0] in _MATRA or token[0] == _VIRAMA:
        return True
    prev_matra = False
    prev_virama = False
    for char in token:
        is_matra = char in _MATRA
        is_virama = char == _VIRAMA
        if is_matra and prev_matra:
            return True
        if is_virama and prev_virama:
            return True
        prev_matra = is_matra
        prev_virama = is_virama
    if token[-1] == _VIRAMA:
        return True
    return False


class GarbledDetector(ABC):
    """Return True when a hypothesis token is unintelligible / collapsed."""

    name: str

    @abstractmethod
    def is_garbled(self, token: str) -> bool:
        """Classify one already-normalized hypothesis token."""


class ExplicitListDetector(GarbledDetector):
    """Only the lexicon ``garbled_tokens`` set. Safe but blind to real CTC errors."""

    name = "explicit"

    def __init__(self, garbled_tokens: Iterable[str]) -> None:
        self.garbled_tokens = {_norm(t) for t in garbled_tokens if _norm(t)}

    def is_garbled(self, token: str) -> bool:
        return _norm(token) in self.garbled_tokens


class CharacterPlausibilityDetector(GarbledDetector):
    """Orthography-only detector, with the explicit list as an override."""

    name = "character"

    def __init__(self, garbled_tokens: Iterable[str] = ()) -> None:
        self._explicit = ExplicitListDetector(garbled_tokens)

    def is_garbled(self, token: str) -> bool:
        norm = _norm(token)
        if self._explicit.is_garbled(norm):
            return True
        return is_character_implausible(norm)


class WordlistOOVDetector(GarbledDetector):
    """Treat every out-of-vocabulary token as garbled. High false-positive risk."""

    name = "wordlist_oov"

    def __init__(self, wordlist: Iterable[str], garbled_tokens: Iterable[str] = ()) -> None:
        self.wordlist = {_norm(t) for t in wordlist if _norm(t)}
        self._explicit = ExplicitListDetector(garbled_tokens)

    def is_garbled(self, token: str) -> bool:
        norm = _norm(token)
        if self._explicit.is_garbled(norm):
            return True
        if not norm:
            return False
        if not self.wordlist:
            return False
        return norm not in self.wordlist


class EditDistanceDetector(GarbledDetector):
    """OOV and farther than ``threshold`` edits from the nearest listed word."""

    name = "edit_distance"

    def __init__(
        self,
        wordlist: Iterable[str],
        garbled_tokens: Iterable[str] = (),
        threshold: int = 2,
    ) -> None:
        self.wordlist = [_norm(t) for t in wordlist if _norm(t)]
        self.wordset = set(self.wordlist)
        self.threshold = int(threshold)
        self._explicit = ExplicitListDetector(garbled_tokens)

    def is_garbled(self, token: str) -> bool:
        norm = _norm(token)
        if self._explicit.is_garbled(norm):
            return True
        if not norm or norm in self.wordset or not self.wordlist:
            return False
        nearest = min_edit_distance(norm, self.wordlist, length_slack=self.threshold + 2)
        if nearest is None:
            return True
        return nearest > self.threshold


class CompositeGarbledDetector(GarbledDetector):
    """Default: explicit list, illegal orthography, or far-OOV."""

    name = "composite"

    def __init__(
        self,
        wordlist: Iterable[str],
        garbled_tokens: Iterable[str] = (),
        threshold: int = 2,
        treat_oov_as_garbled: bool = False,
    ) -> None:
        words = [_norm(t) for t in wordlist if _norm(t)]
        self.wordset = set(words)
        self.wordlist = words
        self.threshold = int(threshold)
        self.treat_oov_as_garbled = treat_oov_as_garbled
        self._explicit = ExplicitListDetector(garbled_tokens)

    def is_garbled(self, token: str) -> bool:
        norm = _norm(token)
        if not norm:
            return False
        if self._explicit.is_garbled(norm):
            return True
        if norm in self.wordset:
            return False
        if is_character_implausible(norm):
            return True
        if not self.wordlist:
            return False
        if self.treat_oov_as_garbled:
            return True
        nearest = min_edit_distance(norm, self.wordlist, length_slack=self.threshold + 2)
        if nearest is None:
            return True
        return nearest > self.threshold


def build_garbled_detector(
    name: str,
    *,
    wordlist: Iterable[str],
    garbled_tokens: Iterable[str] = (),
    threshold: int = 2,
    treat_oov_as_garbled: bool = False,
) -> GarbledDetector:
    """Construct a named detector. Unknown names raise ``ValueError``."""
    key = (name or "composite").strip().lower()
    if key == "explicit":
        return ExplicitListDetector(garbled_tokens)
    if key == "character":
        return CharacterPlausibilityDetector(garbled_tokens)
    if key == "wordlist_oov":
        return WordlistOOVDetector(wordlist, garbled_tokens)
    if key == "edit_distance":
        return EditDistanceDetector(wordlist, garbled_tokens, threshold=threshold)
    if key == "composite":
        return CompositeGarbledDetector(
            wordlist,
            garbled_tokens,
            threshold=threshold,
            treat_oov_as_garbled=treat_oov_as_garbled,
        )
    raise ValueError(
        f"Unknown garbled_detector={name!r}. "
        "Expected explicit, character, wordlist_oov, edit_distance, or composite."
    )
