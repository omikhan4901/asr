"""
Side-by-side Whisper vs Wav2Vec2 evaluation on held-out N-best pools.

1-best for Whisper is rank-0 of the N-best list (the published baseline
definition). 1-best for Wav2Vec2 is the stored greedy CTC transcript when
present, because beam rank-0 can diverge from greedy once KenLM is enabled.
MBR and oracle run on the scored candidate list in both cases.

SWER columns are populated when ``bangla_asr.metrics.swer`` is importable;
otherwise they are left as null so this module can run before the metric lands.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bangla_asr.common.metrics import compute_cer, compute_wer
from bangla_asr.config import MBRConfig
from bangla_asr.rescoring.mbr import MBRSelector
from bangla_asr.rescoring.oracle import evaluate_oracle_ceiling


def _one_best_text(record: Dict[str, Any]) -> str:
    """Select the track-appropriate 1-best transcript.

    Wav2Vec2 JSONL written by this repo stores ``greedy_text``. Whisper JSONL
    from the published pipeline does not; rank 0 remains the baseline 1-best.
    """
    family = str(record.get("model_family", "")).lower()
    greedy = record.get("greedy_text")
    if greedy is not None and family in {"wav2vec2", "wav2vec2_ctc", "ctc"}:
        return str(greedy)
    candidates = record.get("candidates") or []
    if not candidates:
        return ""
    top = min(candidates, key=lambda c: c.get("rank", 0))
    return str(top.get("text", ""))


def _mbr_texts(
    records: Sequence[Dict[str, Any]],
    selector: MBRSelector,
) -> Tuple[List[str], List[str], List[str]]:
    hyps: List[str] = []
    refs: List[str] = []
    one_bests: List[str] = []
    for rec in records:
        candidates = rec.get("candidates") or []
        if not candidates:
            continue
        chosen = selector.select(candidates)
        hyps.append(chosen["text"])
        refs.append(rec.get("ground_truth", ""))
        one_bests.append(_one_best_text(rec))
    return hyps, refs, one_bests


def _optional_corpus_swer(hypotheses: Sequence[str], references: Sequence[str]) -> Optional[float]:
    try:
        from bangla_asr.metrics.swer import corpus_swer
    except ImportError:
        return None
    return float(corpus_swer(list(hypotheses), list(references)))


def evaluate_track(
    records: Sequence[Dict[str, Any]],
    track_name: str,
    mbr_config: Optional[MBRConfig] = None,
) -> Dict[str, Any]:
    """Compute 1-best / MBR / oracle WER (and SWER if available) for one pool."""
    mbr_config = mbr_config or MBRConfig()
    selector = MBRSelector(mbr_config)
    mbr_hyps, refs, one_bests = _mbr_texts(records, selector)
    if not refs:
        raise ValueError(f"Track {track_name!r} has no utterances with candidates.")

    one_best_wer = compute_wer(one_bests, refs)
    one_best_cer = compute_cer(one_bests, refs)
    mbr_wer = compute_wer(mbr_hyps, refs)
    mbr_cer = compute_cer(mbr_hyps, refs)
    oracle = evaluate_oracle_ceiling(list(records))

    headroom = one_best_wer - oracle["wer"]
    recovered = ((one_best_wer - mbr_wer) / headroom * 100.0) if headroom > 0 else 0.0

    return {
        "track": track_name,
        "n_utterances": len(refs),
        "mbr_temperature": mbr_config.temperature,
        "one_best_wer": one_best_wer,
        "one_best_cer": one_best_cer,
        "mbr_wer": mbr_wer,
        "mbr_cer": mbr_cer,
        "oracle_wer": oracle["wer"],
        "oracle_cer": oracle["cer"],
        "oracle_headroom_points": round(headroom * 100.0, 2),
        "mbr_headroom_recovery_pct": round(recovered, 2),
        "one_best_swer": _optional_corpus_swer(one_bests, refs),
        "mbr_swer": _optional_corpus_swer(mbr_hyps, refs),
    }


def compare_acoustic_tracks(
    tracks: Dict[str, Sequence[Dict[str, Any]]],
    mbr_configs: Optional[Dict[str, MBRConfig]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate named N-best pools and optionally write JSON + CSV tables.

    ``mbr_configs`` may supply a per-track temperature. CTC beam scores are
    not on the Whisper log-prob scale; passing a separate temperature is the
    supported way to avoid silently reusing the published 0.5 value.
    """
    mbr_configs = mbr_configs or {}
    results = []
    for name, records in tracks.items():
        cfg = mbr_configs.get(name, MBRConfig())
        results.append(evaluate_track(records, name, cfg))

    summary = {"tracks": results}
    print(format_comparison_table(results))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "acoustic_track_comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        fieldnames = list(results[0].keys()) if results else []
        with open(out / "acoustic_track_comparison.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Saved comparison table to {out}")
    return summary


def format_comparison_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render a fixed-width side-by-side summary for the terminal and logs."""
    lines = [
        "",
        "=" * 88,
        "ACOUSTIC MODEL COMPARISON  (1-best / MBR / oracle)",
        "=" * 88,
    ]
    header = (
        f"{'track':<16} {'n':>5} {'1best WER':>10} {'MBR WER':>10} "
        f"{'oracle':>10} {'headroom':>9} {'recov%':>8} {'1best SWER':>11} {'MBR SWER':>9}"
    )
    lines.append(header)
    lines.append("-" * 88)
    for row in rows:
        def _pct(key: str) -> str:
            val = row.get(key)
            if val is None:
                return "n/a"
            return f"{float(val) * 100:.2f}%"

        lines.append(
            f"{row['track']:<16} {row['n_utterances']:>5} {_pct('one_best_wer'):>10} "
            f"{_pct('mbr_wer'):>10} {_pct('oracle_wer'):>10} "
            f"{row['oracle_headroom_points']:>9.2f} {row['mbr_headroom_recovery_pct']:>8.1f} "
            f"{_pct('one_best_swer'):>11} {_pct('mbr_swer'):>9}"
        )
    lines.append("=" * 88)
    lines.append(
        "Note: CTC beam scores are not on the Whisper log-prob scale. "
        "MBR temperature=0.5 is Whisper-calibrated; set a per-track temperature for CTC."
    )
    lines.append(
        "Note: CTC 1-best uses greedy_text when present (KenLM beam-1 can differ). "
        "Whisper 1-best remains rank-0, matching the published 34.85% path."
    )
    lines.append("=" * 88)
    return "\n".join(lines)
