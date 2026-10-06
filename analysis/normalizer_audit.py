"""E1: recompute 1-best / MBR / oracle WER on the 499A eval N-best under the
current normalizer, and show which weaker normalizer reproduces the published
34.85% 1-best WER.

Usage (from repo root):
    PYTHONPATH=src ./Scripts/python.exe analysis/normalizer_audit.py
"""
import json
import re
import sys

import jiwer

sys.path.insert(0, "src")
from bangla_asr.common.metrics import compute_wer  # noqa: E402
from bangla_asr.config import MBRConfig  # noqa: E402
from bangla_asr.rescoring.mbr import MBRSelector  # noqa: E402
from bangla_asr.rescoring.oracle import evaluate_oracle_headroom  # noqa: E402

NBEST = "data/processed_naive_vad/nbest/eval_nbest_candidates.jsonl"


def basic_punct_only(s: str) -> str:
    # Strips only । , . ? ! -- the rule that reproduces the published 499A numbers.
    return re.sub(r"\s+", " ", re.sub(r"[।,\.?!]", " ", s)).strip()


def main():
    recs = [json.loads(line) for line in open(NBEST, encoding="utf-8")]
    refs = [r["ground_truth"] for r in recs]
    top1 = [min(r["candidates"], key=lambda c: c["rank"])["text"] for r in recs]

    headroom = evaluate_oracle_headroom(recs)
    selector = MBRSelector(MBRConfig())
    mbr = [selector.select(r["candidates"])["text"] for r in recs]

    print(f"n_utterances               {len(recs)}")
    print(f"1-best  (current norm)     {headroom['1best_baseline']['wer']:.4f}")
    print(f"MBR     (current norm)     {compute_wer(mbr, refs):.4f}")
    print(f"oracle  (current norm)     {headroom['oracle_ceiling']['wer']:.4f}")
    print(f"1-best  (no normalization) {compute_wer(top1, refs, normalize=False):.4f}")
    print(f"1-best  (only । , . ? !)    "
          f"{jiwer.wer([basic_punct_only(x) for x in refs], [basic_punct_only(x) for x in top1]):.4f}")


if __name__ == "__main__":
    main()
