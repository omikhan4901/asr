"""
Text normalization and cleaning utilities for Bangla script.
"""

import re
import unicodedata

# Unicode range for Bangla punctuation and common Western punctuation
_PUNCT_REGEX = re.compile(r"[\u0964\u0965,\-–—\.?!:;\"\'‘’“”\(\)\[\]\{\}\<\>/\\@#$%^&*_+=~`|]")
_WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_bangla_text(text: str) -> str:
    """
    Standardizes Bangla text for WER/CER evaluation and metric comparison.
    Recomposes decomposed nukta graphemes (ড়, ঢ়, য়), strips punctuation, and collapses whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))

    # Canonicalize decomposed nukta combinations into single unified Bengali code points
    text = text.replace("\u09a1\u09bc", "\u09dc")  # ড + ় -> ড়
    text = text.replace("\u09a2\u09bc", "\u09dd")  # ঢ + ় -> ঢ়
    text = text.replace("\u09af\u09bc", "\u09df")  # য + ় -> য়

    # Strip punctuation
    text = _PUNCT_REGEX.sub(" ", text)

    # Collapse multiple whitespaces
    text = _WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def clean_text_for_alignment(text: str) -> str:
    """
    Strips punctuation and extraneous characters specifically for RapidFuzz sequential alignment.
    """
    return normalize_bangla_text(text)
