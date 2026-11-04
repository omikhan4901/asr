"""Word-family stem (src/bangla_asr/common/stem.py). Run: PYTHONPATH=src python tests/test_stem.py"""

from bangla_asr.common.stem import family, strip_ending

STRIP = [
    ("ঢাকায়", "ঢাকা"),
    ("ঢাকার", "ঢাকা"),
    ("বাংলাদেশের", "বাংলাদেশ"),
    ("বাংলাদেশে", "বাংলাদেশ"),
    ("বইগুলো", "বই"),
    ("মানুষেরা", "মানুষ"),
    ("ছাত্রদের", "ছাত্র"),
    ("বইটা", "বই"),
    ("বাড়িতে", "বাড়ি"),
    ("সাগর", "সাগর"),
    ("ভাই", "ভাই"),
    ("সে", "সে"),
    ("আমাকে", "আমা"),
]


def main():
    n = 0
    for src, want in STRIP:
        got = strip_ending(src)
        assert got == want, (src, got, want)
        n += 1
    assert family("ছিলো") == family("ছিল"), (family("ছিলো"), family("ছিল"))
    n += 1
    assert family("ঢাকায়") == family("ঢাকা"), (family("ঢাকায়"), family("ঢাকা"))
    n += 1
    print(f"[PASS] stem: {n} cases")


if __name__ == "__main__":
    main()
