"""
Bangla numbers, both directions.

numbers_in     the values of numbers written as words (উনিশশো চুরানব্বই) or digits (১৯৯৪, 1994): checks whether a
               transcript got a number right however it is written, and turns number words into digits for display.
               Spelling variants are accepted through variants.fold (উনিশ / ঊনিশ, এগার / এগারো).
spoken_forms   the ways a digit token can be said (১৯৯৪ -> উনিশশো চুরানব্বই / এক হাজার নয়শো চুরানব্বই, ৫টি -> পাঁচটি,
               ২১শে -> একুশে, ৪৫% -> পঁয়তাল্লিশ শতাংশ / পঁয়তাল্লিশ পার্সেন্ট ...); None when the token is not handled.
               Used to rewrite training text with numbers as words; the audio picks the reading.
"""

import re
from typing import List, Optional, Tuple

from bangla_asr.common.variants import fold

_UNITS = """শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ বিশ
একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আঠাশ ঊনত্রিশ ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ
সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ
ঊনপঞ্চাশ পঞ্চাশ একান্ন বাহান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন ঊনষাট ষাট একষট্টি বাষট্টি তেষট্টি
চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি ঊনসত্তর সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর
আটাত্তর ঊনআশি আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি অষ্টাশি ঊননব্বই নব্বই একানব্বই বিরানব্বই
তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই""".split()
assert len(_UNITS) == 100
# Common alternative forms. "বার" (12) is left out: it usually means "times".
_ALT = {"দু": 2, "ছ": 6, "ছয়": 6, "নয়": 9, "কুড়ি": 20, "উনত্রিশ": 29, "উনচল্লিশ": 39, "উনপঞ্চাশ": 49,
        "উনষাট": 59, "উনসত্তর": 69, "উনআশি": 79, "উননব্বই": 89, "আটাশি": 88, "বিয়াল্লিশ": 42}
_MULT = {"শো": 100, "শ": 100, "শত": 100, "হাজার": 1000, "লাখ": 100_000, "লক্ষ": 100_000, "কোটি": 10_000_000}
_KEY = {fold(w): v for v, w in enumerate(_UNITS)} | {fold(w): v for w, v in _ALT.items()}
_MULT_KEY = {fold(w): v for w, v in _MULT.items()}
_SUFFIXES = ("টা", "টি", "জন", "ের", "এর", "ে", "র", "য়")  # ১৯৯৪-এ, আশির, পাঁচটা, দুজন
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUM = re.compile(r"^[৳$]?([0-9০-৯][0-9০-৯,.]*)")


def _strip(word: str) -> str:
    for s in _SUFFIXES:
        if word.endswith(s) and len(word) > len(s) and (fold(word[:-len(s)]) in _KEY or fold(word[:-len(s)]) in _MULT_KEY):
            return word[:-len(s)]
    return word


def _parts(word: str) -> Optional[List[Tuple[str, int]]]:
    """One word -> [('unit', v) | ('mult', m)] or None. Splits joined hundreds: উনিশশো = উনিশ + শো."""
    w = fold(_strip(word))
    if w in _KEY:
        return [("unit", _KEY[w])]
    if w in _MULT_KEY:
        return [("mult", _MULT_KEY[w])]
    for m, mv in sorted(_MULT_KEY.items(), key=lambda x: -len(x[0])):  # longest suffix first (শত before শ)
        if w.endswith(m) and w[:-len(m)] in _KEY:
            return [("unit", _KEY[w[:-len(m)]]), ("mult", mv)]
    return None


def digit_value(word: str) -> Optional[int]:
    m = _NUM.match(word)
    if not m:
        return None
    s = re.sub(r"[,.]", "", m.group(1).translate(_DIGITS))
    return int(s) if s else None


def _combine(parts: List[Tuple[str, int]]) -> int:
    total, cur = 0, 0
    for kind, v in parts:
        if kind == "unit":
            cur += v
        elif v == 100:
            cur = (cur or 1) * 100
        else:
            total += (cur or 1) * v
            cur = 0
    return total + cur


def numbers_in(words: List[str]) -> List[int]:
    """Values of the numbers in a word sequence. A run of number words is one number (দুই হাজার চব্বিশ = 2024);
    a word with digits is one number. Two units in a row start a new number (এক দুই = 1, 2)."""
    out, run = [], []

    def flush():
        if run:
            out.append(_combine(run))
            run.clear()

    for w in words:
        d = digit_value(w)
        if d is not None:
            flush()
            out.append(d)
            continue
        p = _parts(w)
        if p is None:
            flush()
            continue
        if run and p[0][0] == "unit" and run[-1][0] == "unit" and not (run[-1][1] % 100 == 0 and run[-1][1] >= 100):
            flush()
        run.extend(p)
    flush()
    return out


# ------------------------------------------------------------------ digits -> words
# Conventions from the training text (2026-11-04): hundreds as -শো (একশো 29 vs একশ 24, তিনশো 13 vs তিনশ 7).
_ORDINAL = {1: "প্রথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুর্থ", 5: "পঞ্চম", 6: "ষষ্ঠ", 7: "সপ্তম", 8: "অষ্টম",
            9: "নবম", 10: "দশম"}
_DATE_1_4 = {1: "পহেলা", 2: "দোসরা", 3: "তেসরা", 4: "চৌঠা"}  # ১লা ২রা ৩রা ৪ঠা
_VOWEL_END = re.compile("[অ-ঔা-ৌৗয়]$")  # independent vowel, vowel sign, য়
_TOKEN = re.compile(
    r"^(?P<cur>[৳$])?(?P<num>[0-9০-৯]+(?:,[0-9০-৯]+)*(?:\.[0-9০-৯]+)?)(?P<pct>%)?-?(?P<suf>[ঀ-৿’']*)$")
_RANGE = re.compile(r"^([0-9০-৯]+)[-–]([0-9০-৯]+)$")
_LABEL = re.compile(r"^([ঀ-৿A-Za-z]+)-([0-9০-৯]+)$")  # জিপিএ-৫
_PUNCT_L, _PUNCT_R = "(\"'‘“", ",।?!;:.)\"'’”"
_COUNTERS = ("টি", "টা", "টার", "টায়", "টির", "টাই", "টাও", "টিতে", "টিই", "জন", "জনের", "জনকে", "জনই", "গুণ")


def cardinal(n: int) -> str:
    """Counting form in the Indian system: 1994 -> এক হাজার নয়শো চুরানব্বই, 250000 -> দুই লাখ পঞ্চাশ হাজার."""
    if n < 100:
        return _UNITS[n]
    out = []
    crore, n = divmod(n, 10_000_000)
    if crore:
        out.append(f"{cardinal(crore)} কোটি")
    lakh, n = divmod(n, 100_000)
    if lakh:
        out.append(f"{_UNITS[lakh]} লাখ")
    thousand, n = divmod(n, 1000)
    if thousand:
        out.append(f"{_UNITS[thousand]} হাজার")
    hundred, n = divmod(n, 100)
    if hundred:
        out.append(f"{_UNITS[hundred]}শো")
    if n:
        out.append(_UNITS[n])
    return " ".join(out)


def year(n: int) -> Optional[str]:
    """Year form for 1100-1999 (উনিশশো চুরানব্বই); 2000-2099 are said as counting numbers (দুই হাজার বিশ)."""
    if 1100 <= n <= 1999:
        return f"{_UNITS[n // 100]}শো" + (f" {_UNITS[n % 100]}" if n % 100 else "")
    return None


def digit_by_digit(s: str) -> str:
    return " ".join(_UNITS[int(c)] for c in s.translate(_DIGITS))


def _attach(base: str, suf: str) -> Optional[List[str]]:
    """Case and counter suffixes on the last number word; None if the suffix is not handled."""
    if not suf:
        return [base]
    if suf in _COUNTERS or suf in ("কে", "ও", "তে"):
        return [base + suf]
    vowel = bool(_VOWEL_END.search(base))
    if suf in ("এ", "ে"):
        return [base + "তে", base + "য়ে"] if vowel else [base + "ে"]
    if suf in ("এর", "ের", "র"):
        return [base + "র", base + "য়ের"] if vowel else [base + "ের"]
    return None


def _value_forms(num: str) -> List[str]:
    s = num.translate(_DIGITS)
    if "." in s:
        whole, frac = s.split(".", 1)
        return [f"{cardinal(int(whole.replace(',', '')))} দশমিক {digit_by_digit(frac)}"]
    if s.startswith("0") and len(s) > 1:
        return [digit_by_digit(s)]
    n = int(s.replace(",", ""))
    if n >= 10 ** 9:
        return [digit_by_digit(s)]
    forms = [cardinal(n)]
    if year(n):
        forms.append(year(n))
    if len(s) >= 5 and "," not in s:
        forms.append(digit_by_digit(s))
    return forms


def spoken_forms(token: str, next_word: str = "") -> Optional[List[str]]:
    """Ways to say one whitespace token that contains digits, with its punctuation kept. None = not handled."""
    left = len(token) - len(token.lstrip(_PUNCT_L))
    core = token[left:].rstrip(_PUNCT_R)
    pre, post = token[:left], token[left + len(core):]
    m = _RANGE.match(core)
    if m:
        a, b = (int(x.translate(_DIGITS)) for x in m.groups())
        fa = [cardinal(a)] + ([year(a)] if year(a) else [])
        return [pre + f"{x} {mid}{cardinal(b)}" + post for x in fa for mid in ("থেকে ", "")]
    m = _LABEL.match(core)
    if m:
        return [pre + f"{m.group(1)} {f}" + post for f in _value_forms(m.group(2))]
    m = _TOKEN.match(core)
    if not m:
        return None
    num, suf = m.group("num"), m.group("suf")
    n_plain = None if ("." in num or "," in num) else int(num.translate(_DIGITS))
    if suf in ("ম", "য়", "র্থ", "ষ্ঠ"):  # ১ম ২য় ৪র্থ ৬ষ্ঠ
        return [pre + _ORDINAL[n_plain] + post] if n_plain in _ORDINAL else None
    if suf in ("লা", "রা", "ঠা"):  # ১লা ২রা ৪ঠা
        return [pre + _DATE_1_4[n_plain] + post] if n_plain in _DATE_1_4 else None
    if suf in ("ই", "শে") and n_plain is not None and 5 <= n_plain <= 31:  # ১৫ই পনেরোই, ২১শে একুশে
        w = _UNITS[n_plain]
        return [pre + (w + "ে" if w.endswith("শ") else w + "ই") + post]
    if suf in ("শ’", "শ'", "শ", "শো") and n_plain is not None and n_plain < 100:  # ৫শ’ পাঁচশো (’ = abbreviation mark)
        return [pre + _UNITS[n_plain] + "শো" + post.lstrip("’'")]
    if suf == "তম" and n_plain is not None:
        return [pre + f + "তম" + post for f in _value_forms(num)]
    values = _value_forms(num)
    if n_plain is not None and 1900 <= n_plain <= 2099 and n_plain % 100 and next_word.startswith("সাল") and not suf:
        values.append(_UNITS[n_plain % 100])  # short year in speech: ২০২০ সালে -> বিশ সালে, ১৯৯৪ সালে -> চুরানব্বই সালে
    forms = []
    for f in values:
        attached = _attach(f, suf)
        if attached is None:
            return None
        forms.extend(attached)
    if m.group("pct"):
        forms = [f"{f} শতাংশ" for f in forms] + [f"{f} পার্সেন্ট" for f in forms] + [f"শতকরা {f}" for f in forms]
    if m.group("cur") and not next_word.startswith("টাকা"):
        forms = [f"{f} টাকা" for f in forms]
    return [pre + f + post for f in forms]
