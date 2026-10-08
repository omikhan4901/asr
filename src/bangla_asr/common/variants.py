"""
Bangla spelling-convention folding and rule-based word-error typing (E5, E20, E21, E23).

``fold`` maps a word to a key that is equal for the common spelling variants
of the same word (ছিল/ছিলো, কি/কী, কোন/কোনো, রং/রঙ, ঈদ/ইদ, উৎসব/উত্সব, চাঁদ/চাদ).
It is an *analysis* key only: WER itself is always computed on
``normalize_bangla_text`` output and is not changed by this module.

``classify_alignment`` types every edit of a word alignment. Types, checked in
this order:
  number        either side contains a digit (digits vs number words, formats)
  script        either side contains Latin letters (loanword in English script)
  segmentation  a run of adjacent edits whose words, joined without spaces and
                folded, are equal but split differently (দুপাশে / দু পাশে)
  convention    fold(ref) == fold(hyp): same word, different spelling convention
  vowel_shift   same consonant skeleton, different vowels, and neither word
                is the other plus an ending (নিয়া/নিয়ে, তুমার/তোমার):
                dialect-like forms
  near_miss     folded character distance <= 2 (other small differences)
  substitution  any other substituted word
  deletion / insertion  a word missing from / added to the hypothesis

The rules are proxies; the human labels of the top substitution pairs measure
how precise each type is.
"""

import re
from typing import Dict, List, Sequence

from rapidfuzz.distance import Levenshtein

from bangla_asr.common.metrics import AlignmentOp

TYPES = ["number", "script", "segmentation", "convention", "vowel_shift", "near_miss",
         "substitution", "deletion", "insertion"]
# Types counted as "spelling/dialect variation" in the headline share.
VARIANT_TYPES = ("segmentation", "convention", "vowel_shift")

_DIGIT = re.compile(r"[0-9০-৯]")
_LATIN = re.compile(r"[A-Za-z]")
_VIRAMA = "্"
_FOLD_CHARS = str.maketrans({
    "ী": "ি",  # ী -> ি
    "ূ": "ু",  # ূ -> ু
    "ঈ": "ই",  # ঈ -> ই
    "ঊ": "উ",  # ঊ -> উ
    "ঁ": None,      # chandrabindu
    "‌": None,      # ZWNJ
    "‍": None,      # ZWJ
})
# Stems whose -ো/-ও form is a different word (emphatic particle), not a spelling.
_EMPHATIC_STEMS = {"আর", "এখন", "তখন", "কখন", "যখন", "আবার", "আজ"}
# Dependent vowel signs (া..ৌ, ৗ) and independent vowels (অ..ঔ).
_VOWELS = re.compile(r"[া-ৌৗঅ-ঔ]")


def fold(word: str) -> str:
    """Spelling-convention key of one (already normalized) word."""
    w = word.translate(_FOLD_CHARS)
    w = w.replace("য়", "য")                 # য় -> য
    w = w.replace("ৎ", "ত" + _VIRAMA)       # ৎ -> ত্
    w = re.sub("ঙ(?!্)", "ং", w)       # ঙ -> ং (not in ঙ্গ etc.)
    # Final o-kar: ছিলো/ছিল, হলো/হল, ভালো/ভাল, কোনো/কোন/কোনও are one word. But
    # -ও/-ো is also the emphatic particle ("too", "still"): আমারও != আমার,
    # আরও/আরো ("more") != আর ("and"), এখনো ("still") != এখন ("now").
    stem = w[:-1]
    if w.endswith("ও") and (stem in _EMPHATIC_STEMS or stem == "কোন"):
        w = stem + "ো"                           # আরও -> আরো, কোনও -> কোনো
    if w.endswith("ো") and w[:-1] not in _EMPHATIC_STEMS:
        w = w[:-1]                                    # ছিলো -> ছিল
    return re.sub("্+$", "", w)                  # final hasanta


def skeleton(word: str) -> str:
    """Consonant skeleton of the folded word (vowels removed)."""
    return _VOWELS.sub("", fold(word))


def pair_type(ref: str, hyp: str) -> str:
    """Type of one substituted word pair (ref != hyp)."""
    if _DIGIT.search(ref) or _DIGIT.search(hyp):
        return "number"
    if _LATIN.search(ref) or _LATIN.search(hyp):
        return "script"
    fr, fh = fold(ref), fold(hyp)
    if fr == fh:
        return "convention"
    sr = skeleton(ref)
    # A changed vowel, not an added/dropped ending (আর/আরো, কর/করে are word changes).
    if sr and sr == skeleton(hyp) and not (fr.startswith(fh) or fh.startswith(fr)):
        return "vowel_shift"
    if Levenshtein.distance(fr, fh) <= 2:
        return "near_miss"
    return "substitution"


def align_words(ref: Sequence[str], hyp: Sequence[str]) -> List[AlignmentOp]:
    """Minimum-edit word alignment with a linguistic tie-break, left to right.

    Among all alignments with the minimum number of edits (so the edit count is
    exactly the WER numerator), keep the most exact matches, then pair the most
    similar words. Plain Levenshtein backtraces break ties arbitrarily and turn
    "দু পাশে ... ছিল" vs "দুপাশে ... ছিলো" into a chain of unrelated substitutions.
    """
    n, m = len(ref), len(hyp)
    # cost = (edits, -matches, dissimilarity) packed into one integer.
    E, M = 10 ** 13, 10 ** 8  # keeps the three keys apart for up to ~10^4 words

    def sub_cost(a, b):
        return 0 if a == b else E + int(1000 * Levenshtein.normalized_distance(fold(a), fold(b)))

    gap = E + 1000
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        r, row, prev = ref[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1):
            c = sub_cost(r, hyp[j - 1])
            diag = prev[j - 1] + (c - M if c == 0 else c)
            row[j] = min(diag, prev[j] + gap, row[j - 1] + gap)
    ops, i, j = [], n, m
    while i or j:
        if i and j:
            c = sub_cost(ref[i - 1], hyp[j - 1])
            if D[i][j] == D[i - 1][j - 1] + (c - M if c == 0 else c):
                ops.append(AlignmentOp("equal" if c == 0 else "substitute", ref[i - 1], hyp[j - 1], i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i and D[i][j] == D[i - 1][j] + gap:
            ops.append(AlignmentOp("delete", ref[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(AlignmentOp("insert", None, hyp[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _segmentation_spans(ops) -> List[range]:
    """Index ranges of `ops` that are re-spacings of the same folded letters."""
    spans, k = [], 0
    while k < len(ops):
        if ops[k].op == "equal":
            k += 1
            continue
        end = k
        while end < len(ops) and ops[end].op != "equal":
            end += 1
        # Longest sub-window first inside this run of edits.
        s = k
        while s < end:
            found = None
            for e in range(end, s + 1, -1):
                win = ops[s:e]
                r = [o.ref_token for o in win if o.ref_token is not None]
                h = [o.hyp_token for o in win if o.hyp_token is not None]
                if r and h and len(r) != len(h) and "".join(map(fold, r)) == "".join(map(fold, h)):
                    found = e
                    break
            if found:
                spans.append(range(s, found))
                s = found
            else:
                s += 1
        k = end
    return spans


def classify_alignment(ref: Sequence[str], hyp: Sequence[str]) -> List[Dict]:
    """One dict per edit: op, type, ref_index, hyp_index, ref_word, hyp_word.

    The number of returned edits equals the word edit distance, so type counts
    add up to the WER numerator exactly.
    """
    ops = align_words(ref, hyp)
    seg = set()
    for span in _segmentation_spans(ops):
        seg.update(span)
    out = []
    for k, o in enumerate(ops):
        if o.op == "equal":
            continue
        if k in seg:
            t = "segmentation"
        elif o.op == "substitute":
            t = pair_type(o.ref_token, o.hyp_token)
        else:
            t = "deletion" if o.op == "delete" else "insertion"
        out.append({"op": o.op, "type": t, "ref_index": o.ref_index, "hyp_index": o.hyp_index,
                    "ref_word": o.ref_token or "", "hyp_word": o.hyp_token or ""})
    return out


def fold_text(text: str) -> str:
    """Fold every word of a normalized text (for convention-insensitive WER)."""
    return " ".join(fold(w) for w in text.split())
