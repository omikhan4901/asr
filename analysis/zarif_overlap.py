#!/usr/bin/env python3
"""
Direct contamination check of zarifmahir21/whisper-medium-bangla: its model card names its training data,
zarifmahir21/bengali-asr-chunked (public). Compare that dataset's file names and transcripts with our DL Sprint eval
clips: (1) which DL Sprint recordings it contains, (2) the share of 8-word passages of our eval transcripts that appear
verbatim in its training transcripts, with FLEURS test as a control.

Step 1 needs pyarrow (with NumPy >= 2; use a separate environment): it reads only the file_name, transcript and
duration columns (a few MB, no audio) into data/external/zarif_bengali_asr_chunked_meta.csv.
  python -I analysis/zarif_overlap.py --fetch
  PYTHONPATH=src .venv/bin/python analysis/zarif_overlap.py
Output: results/contamination/zarif_overlap.json
"""

import argparse
import json
import sys
from pathlib import Path

META = Path("data/external/zarif_bengali_asr_chunked_meta.csv")


def fetch():
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    parts = []
    for f in sorted(fs.glob("datasets/zarifmahir21/bengali-asr-chunked/data/*.parquet")):
        with fs.open(f, "rb", block_size=2 ** 20) as fh:
            parts.append(pq.ParquetFile(fh).read(columns=["file_name", "transcript", "duration"]).to_pandas())
    META.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(META, index=False)


def analyse():
    import pandas as pd
    sys.path.insert(0, "src")
    from bangla_asr.common.text import normalize_bangla_text as N
    z = pd.read_csv(META)
    ev = pd.read_csv("data/splits_fa/eval.csv", encoding="utf-8-sig")
    rec = set(z.file_name.str.extract(r"(train_\d{3})")[0].dropna())
    corpus = " ".join(N(str(t)) for t in z.transcript)

    def found(texts, n=8):
        hit = tot = 0
        for t in texts:
            w = N(str(t)).split()
            for i in range(0, len(w) - n + 1, n):
                tot += 1
                hit += " ".join(w[i:i + n]) in corpus
        return hit, tot

    e_hit, e_tot = found(ev.text)
    f = pd.read_csv("data/external/fleurs/test.csv", encoding="utf-8-sig")
    f_hit, f_tot = found(f.text)
    out = {"dataset": "zarifmahir21/bengali-asr-chunked", "clips": len(z), "hours": round(z.duration.sum() / 3600, 1),
           "dlsprint_recordings_in_dataset": len(rec),
           "our_eval_recordings_in_dataset": len(set(ev.recording) & rec), "our_eval_recordings": ev.recording.nunique(),
           "eval_8gram_passages_found": e_hit, "eval_8gram_passages": e_tot,
           "fleurs_8gram_passages_found": f_hit, "fleurs_8gram_passages": f_tot}
    Path("results/contamination/zarif_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    (fetch if ap.parse_args().fetch else analyse)()
