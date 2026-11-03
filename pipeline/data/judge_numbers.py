#!/usr/bin/env python3
"""
The released model as the judge of what number was said (number fix, v1.1).

MMS forced alignment cannot hear long number words (82% of years: <= 2 characters decoded), so it cannot tell
উনিশশো চুরানব্বই from চুরানব্বই or from a skipped number. The released model can: in words it writes numbers well (the
bug is its digit mode). This transcribes every clip of the alignment report with digit tokens and punctuation-only
tokens blocked, so the model has to write numbers as words. verbalize_numbers.py mix --judge then keeps, for each digit
token, the reading the model wrote (and drops the clip when the model wrote another number or none).

Using the model to select its own TRAINING data is allowed; evaluation data is never filtered this way.

  PYTHONPATH=src python pipeline/data/judge_numbers.py --manifest data/splits_fa/matrix/m5-300.csv \
      --report data/splits_fa/matrix/numbers_report.csv --out data/splits_fa/matrix/numbers_judge.csv \
      --model OmiKhan/bangla-asr-whisper-medium [--limit 50]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verbalize_numbers import load_audio  # noqa: E402

DIGIT = re.compile(r"[0-9০-৯]")
LETTER = re.compile(r"[ঀ-৿A-Za-z]")


def blocked_ids(tok) -> list:
    """Tokens with a digit, or with only punctuation/symbols. Partial-byte tokens (decode to U+FFFD) and pure
    whitespace stay allowed: Bangla letters are built from them."""
    ids = []
    for i in range(tok.convert_tokens_to_ids("<|endoftext|>")):
        s = tok.decode([i])
        if DIGIT.search(s) or ("�" not in s and s.strip() and not LETTER.search(s)):
            ids.append(i)
    return ids


def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="OmiKhan/bangla-asr-whisper-medium")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rep = pd.read_csv(args.report, encoding="utf-8-sig")
    ids = rep.loc[rep["status"] == "ok", "utt_id"]
    man = pd.read_csv(args.manifest, encoding="utf-8-sig").set_index("utt_id")
    rows = man.loc[ids].reset_index()
    if args.limit:
        rows = rows.head(args.limit)
    out = Path(args.out)
    done = pd.read_csv(out, encoding="utf-8-sig") if out.exists() else pd.DataFrame(columns=["utt_id", "judge"])
    rows = rows[~rows["utt_id"].isin(set(done["utt_id"]))]  # resumable: finished clips are kept
    print(f"{len(rows)} clips to judge ({len(done)} already done)", flush=True)

    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    suppress = list(model.generation_config.suppress_tokens or []) + blocked_ids(proc.tokenizer)
    print(f"blocked tokens: {len(suppress)}", flush=True)

    results, t0 = [], time.time()
    for k in range(0, len(rows), args.batch_size):
        batch = rows.iloc[k:k + args.batch_size]
        audio = [load_audio(p) for p in batch["audio_path"]]
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            gen = model.generate(feats, language="bn", task="transcribe", suppress_tokens=suppress,
                                 max_new_tokens=440)
        texts = proc.tokenizer.batch_decode(gen, skip_special_tokens=True)
        results += [{"utt_id": u, "judge": t.strip()} for u, t in zip(batch["utt_id"], texts)]
        if (k // args.batch_size) % 25 == 0 or k + args.batch_size >= len(rows):
            pd.concat([done, pd.DataFrame(results)]).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"{len(done) + len(results)} judged, {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
