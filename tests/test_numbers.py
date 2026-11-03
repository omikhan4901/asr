"""Bangla number reading (src/bangla_asr/common/numbers.py). Run: PYTHONPATH=src python tests/test_numbers.py"""

from bangla_asr.common.numbers import cardinal, numbers_in, spoken_forms
from bangla_asr.common.text import normalize_bangla_text

CASES = [
    ("সর্বশেষ উনিশশো চুরানব্বই সালে", [1994]),
    ("সর্বশেষ ১৯৯৪ সালে", [1994]),
    ("সর্বশেষ 1994 সালে", [1994]),
    ("দুই হাজার চব্বিশ সালের নির্বাচন", [2024]),
    ("একাত্তরের মুক্তিযুদ্ধ", [71]),
    ("বাতাসটি সত্তর কিলোমিটার", [70]),
    ("আট লাখ লাল ফৌজ", [800000]),
    ("৳2000 টাকা", [2000]),
    ("ঊনত্রিশ বছর বয়সী", [29]),
    ("উনত্রিশ বছর বয়সী", [29]),
    ("পাঁচশো টাকা আর তিনশ টাকা", [500, 300]),
    ("দুজন লোক", [2]),
    ("এক দুই তিন", [1, 2, 3]),
    ("তিন কোটি পঞ্চাশ লাখ", [35000000]),
    ("আমি বাড়ি যাব", []),
]


SPOKEN = [  # token, next word, expected forms (first = the default)
    ("১৯৯৪", "সালে", ["এক হাজার নয়শো চুরানব্বই", "উনিশশো চুরানব্বই", "চুরানব্বই"]),
    ("2020", "সালে", ["দুই হাজার বিশ", "বিশ"]),
    ("2020", "", ["দুই হাজার বিশ"]),
    ("৭০", "", ["সত্তর"]),
    ("৫টি", "", ["পাঁচটি"]),
    ("২১শে", "ফেব্রুয়ারি", ["একুশে"]),
    ("১৫ই", "আগস্ট", ["পনেরোই"]),
    ("১লা", "বৈশাখ", ["পহেলা"]),
    ("৩য়", "", ["তৃতীয়"]),
    ("২৫তম", "", ["পঁচিশতম"]),
    ("৫শ’", "", ["পাঁচশো"]),
    ("৪৫%", "", ["পঁয়তাল্লিশ শতাংশ", "পঁয়তাল্লিশ পার্সেন্ট", "শতকরা পঁয়তাল্লিশ"]),
    ("৳2000", "", ["দুই হাজার টাকা"]),
    ("৳2000", "টাকা", ["দুই হাজার"]),
    ("২০২০-এ", "", ["দুই হাজার বিশে"]),
    ("৯৯-এর", "", ["নিরানব্বইর", "নিরানব্বইয়ের"]),
    ("৩.৫", "", ["তিন দশমিক পাঁচ"]),
    ("১,৫০০", "", ["এক হাজার পাঁচশো", "পনেরোশো"]),  # both are said (পনেরোশো টাকা)
    ("৯-১০", "", ["নয় থেকে দশ", "নয় দশ"]),
    ("জিপিএ-৫", "", ["জিপিএ পাঁচ"]),
    ("৫০০০,", "", ["পাঁচ হাজার,"]),
    ("১০:৩০", "", None),
]


def main():
    for text, want in CASES:
        got = numbers_in(normalize_bangla_text(text).split())
        assert got == want, (text, got, want)
    for token, nxt, want in SPOKEN:
        got = spoken_forms(token, nxt)
        assert got == want, (token, got, want)
    # round trip: every counting form reads back to its value
    for n in [0, 7, 19, 99, 100, 101, 999, 1000, 1994, 2024, 12345, 250000, 3500000, 10000000, 123456789]:
        assert numbers_in(cardinal(n).split()) == [n], (n, cardinal(n))
    print(f"[PASS] numbers: {len(CASES)} reading cases, {len(SPOKEN)} spoken-form cases, round trip")


if __name__ == "__main__":
    main()
