"""
Tests for spelling-variant folding, error typing, the bootstrap helper and run records.
Run: PYTHONPATH=src ./Scripts/python.exe tests/test_variants.py
"""

import io
import json
import random
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from rapidfuzz.distance import Levenshtein

from bangla_asr.common.provenance import RunRecord
from bangla_asr.common.stats import bootstrap_ratio
from bangla_asr.common.variants import align_words, classify_alignment, pair_type

# 1. Pair types: spelling conventions vs real word changes
cases = {
    ("ছিল", "ছিলো"): "convention", ("কি", "কী"): "convention", ("কোন", "কোনো"): "convention",
    ("কোন", "কোনও"): "convention", ("রং", "রঙ"): "convention", ("ঈদ", "ইদ"): "convention",
    ("উৎসব", "উত্সব"): "convention", ("চাঁদ", "চাদ"): "convention", ("আরও", "আরো"): "convention",
    ("এখনও", "এখনো"): "convention", ("মত", "মতো"): "convention",
    # emphatic particle -ও/-ো is a different word
    ("আর", "আরো"): "near_miss", ("এখন", "এখনো"): "near_miss", ("আমার", "আমারও"): "near_miss",
    ("নিয়ে", "নিয়া"): "vowel_shift", ("তোমার", "তুমার"): "vowel_shift", ("করে", "কইরা"): "vowel_shift",
    ("মানুষ", "মানুস"): "near_miss", ("বাংলা", "বাংলাদেশ"): "substitution",
    ("২০", "বিশ"): "number", ("ফোন", "phone"): "script",
}
for (r, h), want in cases.items():
    got = pair_type(r, h)
    assert got == want, f"{r}/{h}: {got} != {want}"
print(f"[PASS] {len(cases)} pair types.")

# 2. Segmentation + convention + insertion in one utterance
ed = classify_alignment("রাস্তার দু পাশে গাছ ছিল".split(), "রাস্তার দুপাশে গাছ ছিলো আর".split())
assert [e["type"] for e in ed] == ["segmentation", "segmentation", "convention", "insertion"], ed
print("[PASS] segmentation run detected.")

# 3. Typed edits always add up to the word edit distance (the WER numerator)
rng = random.Random(0)
vocab = ["ছিল", "ছিলো", "দু", "পাশে", "দুপাশে", "আর", "আরো", "গাছ", "কি", "কী", "নিয়ে", "নিয়া"]
for _ in range(500):
    ref = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
    hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
    assert len(classify_alignment(ref, hyp)) == Levenshtein.distance(ref, hyp), (ref, hyp)
    ops = align_words(ref, hyp)
    assert [o.ref_token for o in ops if o.ref_token is not None] == ref
    assert [o.hyp_token for o in ops if o.hyp_token is not None] == hyp
print("[PASS] edit counts equal Levenshtein distance on 500 random pairs.")

# 4. Cluster bootstrap of a ratio
b = bootstrap_ratio(["a", "a", "b", "c"], [1, 1, 2, 0], [10, 10, 10, 10], n_boot=200)
assert abs(b["value"] - 0.1) < 1e-12 and b["n_groups"] == 3 and b["ci95"][0] <= 0.1 <= b["ci95"][1]
print("[PASS] bootstrap_ratio.")

# 5. Run records: status, hashes, and appended resumes
with tempfile.TemporaryDirectory() as d:
    rec_path, out = Path(d, "run_info.json"), Path(d, "out.txt")
    with RunRecord(rec_path) as rec:
        out.write_text("x", encoding="utf-8")
        rec.outputs = [out]
    try:
        with RunRecord(rec_path):
            raise ValueError("boom")
    except ValueError:
        pass
    runs = json.loads(rec_path.read_text(encoding="utf-8"))["runs"]
    assert [r["status"] for r in runs] == ["ok", "failed: ValueError"]
    assert runs[0]["outputs"][str(out)] == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
print("[PASS] run records.")
print("All variant tests passed.")
