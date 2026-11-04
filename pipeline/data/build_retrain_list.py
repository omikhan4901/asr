#!/usr/bin/env python3
"""
Build the retrain-v2 training list (PLAN §13 / PROGRESS §97).

The M5-300 list with the number fix (a digit clip whose judge reading is confirmed is rewritten
as those spoken words; every other digit clip is dropped), plus extra Kathbath / Shrutilipi clips
that are not already in it, minus every row whose normalized text equals a sentence of DL Sprint
eval/dev, FLEURS test/validation, Common Voice test/dev, or Kathbath valid.

Inputs:
  --m5_300           M5-300 manifest (default data/backup/modal_friend/manifests/matrix/m5-300.csv)
  --numbers_report   verbalize_numbers.py align report; only status == ok rows have judge text
  --numbers_judge    judge_numbers.py output (utt_id, judge)
  --extra            new-set manifests (utt_id, audio_path, text, group, duration); source is the file stem
  --out              the list (default data/matrix/retrain-v2.csv); counts are written to <out>.json

Stop rules (§97), both booleans in the report: extra Kathbath + Shrutilipi hours after the filters
are at least 50, and the contamination filter drops at most 5% of rows.

  PYTHONPATH=src python pipeline/data/build_retrain_list.py \
      --numbers_report data/splits_fa/matrix/numbers_report.csv \
      --numbers_judge data/splits_fa/matrix/numbers_judge.csv \
      --extra data/splits_fa/matrix/kathbath-full.csv data/splits_fa/matrix/shrutilipi-150.csv \
      --out data/matrix/retrain-v2.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verbalize_numbers import DIGIT, choose_by_judge  # noqa: E402

from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

COLUMNS = ["utt_id", "audio_path", "text", "group", "duration", "source"]
# Normalized sentences of these files are contamination. Names are the count keys.
TEST_SETS = [
    ("splits_fa-eval", "data/splits_fa/eval.csv"),
    ("splits_fa-dev", "data/splits_fa/dev.csv"),
    ("fleurs-test", "data/external/fleurs/test.csv"),
    ("fleurs-validation", "data/external/fleurs/validation.csv"),
    ("cv-test", "data/external/cv/test.csv"),
    ("cv-dev", "data/external/cv/dev.csv"),
    ("kathbath-valid", "data/external/kathbath/valid.csv"),
]


def _norm(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return normalize_bangla_text(str(text))


def _hours(duration) -> float:
    return float(duration.sum()) / 3600 if len(duration) else 0.0


def _corpus(source: str) -> str:
    """kathbath-full and kathbath-train are the same corpus: the prefix before the first '-'."""
    return str(source).split("-", 1)[0]


def _key(group, text, duration) -> tuple:
    return (str(group), _norm(text), round(float(duration), 2))


def _load_test_sentences() -> dict[str, set[str]]:
    sentences = {}
    for name, path in TEST_SETS:
        text = pd.read_csv(path, encoding="utf-8-sig", usecols=["text"])["text"]
        norms = {_norm(t) for t in text}
        norms.discard("")
        sentences[name] = norms
    return sentences


def _number_fix(m5: pd.DataFrame, report_path: str, judge_path: str) -> tuple[pd.DataFrame, dict]:
    """Rewrite judge-confirmed digit clips; drop every other row whose text contains a digit."""
    report = pd.read_csv(report_path, encoding="utf-8-sig", usecols=["utt_id", "status"])
    judge = pd.read_csv(judge_path, encoding="utf-8-sig", usecols=["utt_id", "judge"])
    report = report.merge(judge, on="utt_id", how="left")
    judge_text = {}
    for utt_id, status, spoken in zip(report["utt_id"], report["status"], report["judge"]):
        if status == "ok" and isinstance(spoken, str):
            judge_text[utt_id] = spoken

    n_digit = n_verbalized = n_dropped = 0
    drop_idx = []
    new_text = {}
    for row in m5.itertuples(index=True):
        text = "" if pd.isna(row.text) else str(row.text)
        if not DIGIT.search(text):
            continue
        n_digit += 1
        spoken = judge_text.get(row.utt_id)
        verbal, status = choose_by_judge(text, spoken) if spoken is not None else ("", "no_judge")
        if status == "ok":
            n_verbalized += 1
            new_text[row.Index] = verbal
        else:
            n_dropped += 1
            drop_idx.append(row.Index)

    kept = m5.drop(index=drop_idx).copy()
    if new_text:
        kept["text"] = [new_text.get(i, t) for i, t in zip(kept.index, kept["text"])]
    counts = {"digit_rows": n_digit, "verbalized": n_verbalized, "dropped": n_dropped}
    return kept, counts


def _add_extra(m5_original: pd.DataFrame, paths: list[str]) -> tuple[list[pd.DataFrame], dict]:
    """Clips whose (group, normalized text, duration) key is not already in the same M5-300 corpus."""
    keys: dict[str, set[tuple]] = {}
    corpus = m5_original["source"].astype(str).map(_corpus)
    for name, idx in corpus.groupby(corpus).groups.items():
        part = m5_original.loc[idx]
        keys[name] = set(_key(g, t, d) for g, t, d in zip(part["group"], part["text"], part["duration"]))

    frames, counts = [], {}
    for path in paths:
        extra = pd.read_csv(path, encoding="utf-8-sig")
        source = Path(path).stem
        seen = keys.get(_corpus(source), set())
        duplicates = 0
        keep = []
        for row in extra.itertuples(index=False):
            duplicate = _key(row.group, row.text, row.duration) in seen
            duplicates += int(duplicate)
            keep.append(not duplicate)
        added = extra.loc[pd.Series(keep, index=extra.index), ["utt_id", "audio_path", "text", "group", "duration"]].copy()
        added["source"] = source
        added["utt_id"] = source + "_" + added["utt_id"].astype(str)  # unique even if the HF data changed since the first pull
        added["__extra"] = True
        frames.append(added)
        counts[source] = {
            "rows_in": int(len(extra)),
            "duplicates_dropped": int(duplicates),
            "rows_added": int(len(added)),
            "hours_added": round(_hours(added["duration"]), 3),
        }
    return frames, counts


def _contamination(frame: pd.DataFrame, test_sentences: dict[str, set[str]]) -> tuple[pd.Series, dict]:
    """Drop each training row once. Count it for every test set whose sentence it matches."""
    norms = frame["text"].map(_norm)
    blocked = set().union(*test_sentences.values()) if test_sentences else set()
    drop = norms.map(lambda n: bool(n) and n in blocked)
    per_set = {}
    for name, sentences in test_sentences.items():
        per_set[name] = int(norms.map(lambda n, s=sentences: bool(n) and n in s).sum())
    per_source = {str(s): 0 for s in frame["source"].astype(str).unique()}
    for source, n in frame.loc[drop, "source"].astype(str).value_counts().items():
        per_source[str(source)] = int(n)
    return drop, {"dropped": int(drop.sum()), "per_source": per_source, "per_test_set": per_set}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m5_300", default="data/backup/modal_friend/manifests/matrix/m5-300.csv")
    p.add_argument("--numbers_report", required=True)
    p.add_argument("--numbers_judge", required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--out", default="data/matrix/retrain-v2.csv")
    args = p.parse_args()

    m5 = pd.read_csv(args.m5_300, encoding="utf-8-sig")
    kept, number_fix = _number_fix(m5, args.numbers_report, args.numbers_judge)
    kept["__extra"] = False
    extra_frames, extra_counts = _add_extra(m5, args.extra)
    combined = pd.concat([kept[COLUMNS + ["__extra"]], *extra_frames], ignore_index=True)

    test_sentences = _load_test_sentences()
    drop, contamination = _contamination(combined, test_sentences)
    final = combined.loc[~drop].reset_index(drop=True)
    before = len(combined)
    share = (contamination["dropped"] / before) if before else 0.0
    extra_hours = _hours(final.loc[final["__extra"], "duration"])

    out = final[COLUMNS]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    info = {
        "m5_300_rows": int(len(m5)),
        "digit_rows": number_fix["digit_rows"],
        "verbalized": number_fix["verbalized"],
        "dropped": number_fix["dropped"],
        "extra": extra_counts,
        "contamination_dropped": contamination["dropped"],
        "contamination_per_source": contamination["per_source"],
        "contamination_per_test_set": contamination["per_test_set"],
        "final_rows": int(len(out)),
        "hours_per_source": {str(s): round(_hours(g), 3) for s, g in out.groupby("source", sort=True)["duration"]},
        "total_hours": round(_hours(out["duration"]), 3),
        "extra_hours_after_filters": round(extra_hours, 3),
        "contamination_drop_share": round(share, 6),
        "extra_hours_after_filters >= 50": extra_hours >= 50,
        "contamination_drop_share <= 0.05": share <= 0.05,
    }
    Path(str(out_path) + ".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
