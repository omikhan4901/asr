"""Retrieval by word family (spelling variants + inflections), for Rafat's review.

The ending list is fixed; it was not tuned on retrieval results.
"""

import re

from bangla_asr.common.variants import fold

ENDINGS = (
    "গুলোকে", "গুলোতে", "গুলোর", "গুলোয়", "গুলো", "গুলি",
    "দেরকে", "দের", "েরা", "রা",
    "টাকে", "টিকে", "টাতে", "টিতে", "টার", "টির", "টা", "টি",
    "খানা", "খানি", "য়ের", "এর", "ের", "কে", "রে", "েতে", "তে", "য়", "ে", "র",
)

_BEFORE_R = re.compile(r"[া-ৌঅ-ঔ]$")


def _canon(s: str) -> str:
    return s.replace("\u09af\u09bc", "\u09df")


# Longest first, as written; য় is stored decomposed in this list and composed for matching
# so both spec test strings and normalize_bangla_text output hit the same ending.
_SORTED = tuple(_canon(e) for e in sorted(ENDINGS, key=len, reverse=True))


def strip_ending(word: str) -> str:
    """Remove the longest listed inflectional ending, leaving at least 2 characters."""
    w = _canon(word)
    for ending in _SORTED:
        if not w.endswith(ending):
            continue
        stem = w[: -len(ending)]
        if len(stem) < 2:
            continue
        if ending == "র" and not _BEFORE_R.search(stem):
            continue
        return stem
    return word


def family(word: str) -> str:
    """Spelling-convention key of the word with one inflectional ending stripped."""
    return fold(strip_ending(word))
