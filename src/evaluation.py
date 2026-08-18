"""
WER/CER computation, 1-best baseline evaluation, and oracle-WER ceiling check
over the N-best candidates produced by inference.py.
"""

import json
import argparse
import re
from pathlib import Path

import pandas as pd
import jiwer


def normalize_bangla_text(text: str) -> str:
    """
    Minimal shared normalization applied to BOTH hypothesis and ground truth
    before WER computation, so comparisons are fair. Mirrors the cleanup used
    during alignment (preprocess_chunk_align.py's clean_text_for_alignment),
    but kept separate here since WER scoring and fuzzy alignment have
    slightly different needs (alignment strips more aggressively for
    matching; WER scoring should stay close to "real" output).
    """
    text = str(text).strip()
    text = re.sub(r"[।,\-.?!:;\"'()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    hyps = [normalize_bangla_text(h) for h in hypotheses]
    refs = [normalize_bangla_text(r) for r in references]
    # Drop pairs where reference is empty — jiwer errors on empty refs, and
    # an empty ground-truth span isn't a meaningful data point anyway.
    pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]
    if not pairs:
        return float("nan")
    hyps, refs = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    hyps = [normalize_bangla_text(h) for h in hypotheses]
    refs = [normalize_bangla_text(r) for r in references]
    pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]
    if not pairs:
        return float("nan")
    hyps, refs = zip(*pairs)
    return jiwer.cer(list(refs), list(hyps))


def load_nbest_records(nbest_jsonl: str) -> list[dict]:
    records = []
    with open(nbest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_1best_baseline(records: list[dict]) -> dict:
    """
    Rank-0 candidate from each record = Whisper's own top beam pick.
    This is your primary baseline number to compare every rescoring method
    against (target: ~34.07% WER on the reference checkpoint per prior work).
    """
    hyps, refs = [], []
    for rec in records:
        top_candidate = min(rec["candidates"], key=lambda c: c["rank"])
        hyps.append(top_candidate["text"])
        refs.append(rec["ground_truth"])

    return {
        "wer": compute_wer(hyps, refs),
        "cer": compute_cer(hyps, refs),
        "n_utterances": len(records),
    }


def evaluate_oracle(records: list[dict]) -> dict:
    """
    For each utterance, picks whichever candidate has the LOWEST WER against
    ground truth. This is the theoretical ceiling for any rescoring method —
    it tells you how much headroom actually exists in the N-best pool.

    If oracle WER isn't meaningfully better than 1-best WER, the beams are
    too similar/undiverse and no rescoring method will help much — worth
    checking this BEFORE investing time in MBR/LM rescoring.
    """
    hyps, refs = [], []
    per_utt_oracle_wer = []

    for rec in records:
        ref = rec["ground_truth"]
        best_wer = float("inf")
        best_text = rec["candidates"][0]["text"]

        for cand in rec["candidates"]:
            w = compute_wer([cand["text"]], [ref])
            if not pd.isna(w) and w < best_wer:
                best_wer = w
                best_text = cand["text"]

        hyps.append(best_text)
        refs.append(ref)
        per_utt_oracle_wer.append(best_wer if best_wer != float("inf") else None)

    return {
        "wer": compute_wer(hyps, refs),
        "cer": compute_cer(hyps, refs),
        "n_utterances": len(records),
        "per_utt_oracle_wer": per_utt_oracle_wer,
    }


def run_baseline_and_oracle_eval(nbest_jsonl: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_nbest_records(nbest_jsonl)
    print(f"📊 Loaded {len(records)} utterances with N-best candidates.")

    baseline = evaluate_1best_baseline(records)
    oracle = evaluate_oracle(records)

    results = {
        "1best_baseline": {"wer": baseline["wer"], "cer": baseline["cer"]},
        "oracle_upper_bound": {"wer": oracle["wer"], "cer": oracle["cer"]},
        "oracle_gap_wer_points": (baseline["wer"] - oracle["wer"]) * 100,
        "n_utterances": baseline["n_utterances"],
    }

    results_path = output_path / "baseline_oracle_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS: 1-BEST BASELINE vs. ORACLE UPPER BOUND")
    print("=" * 60)
    print(f"1-best WER:        {baseline['wer']*100:.2f}%")
    print(f"1-best CER:        {baseline['cer']*100:.2f}%")
    print(f"Oracle WER:        {oracle['wer']*100:.2f}%")
    print(f"Oracle CER:        {oracle['cer']*100:.2f}%")
    print(f"Oracle gap:        {results['oracle_gap_wer_points']:.2f} WER points")
    print(f"N utterances:      {baseline['n_utterances']}")
    print("=" * 60)

    if results["oracle_gap_wer_points"] < 2.0:
        print("⚠️  WARNING: Oracle gap is small (<2 WER points). This means your")
        print("   N-best candidates are very similar to each other — rescoring")
        print("   methods likely won't help much. Consider increasing num_beams")
        print("   or enabling diverse beam search (num_beam_groups + diversity_penalty).")

    print(f"\nSaved to: {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest_jsonl", type=str, default="data/processed/nbest/nbest_candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/processed/results")
    args = parser.parse_args()

    run_baseline_and_oracle_eval(args.nbest_jsonl, args.output_dir)