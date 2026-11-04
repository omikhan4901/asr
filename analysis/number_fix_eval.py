#!/usr/bin/env python3
"""
Number-fix evaluation (PROGRESS §87-92).

The released model (v1.0) was trained on text that mixes digits and number words, and in digit
mode it often writes a sound-alike value. v1.1 and v1.2 were fine-tuned to write numbers as words.
This script turns the saved predictions into three tables:

  dev_split.csv        DL Sprint dev WER for v1.0 (v1.1/step_0), v1.1 (step_325) and v1.2
                       (v1.2/step_449): clips whose reference has no digit, clips with a digit,
                       all clips, and clips whose hypothesis repeats one word 5 or more times.
  test_drift.csv       paired WER of v1.2 minus v1.0 on test clips with no digit in the reference
                       (splits_fa-eval, fleurs-test, cv-test, kathbath-valid), in percentage points,
                       from analysis/paired_compare.py.
  number_accuracy.csv  reference digit values found in the hypothesis (numbers_in), DL Sprint eval
                       and FLEURS, v1.0 (final_a100) and v1.2 (numbers_v2).

WER uses normalize_bangla_text and rapidfuzz Levenshtein distance on word lists (corpus edits /
reference words). A reference token counts toward number accuracy when it contains a digit and
digit_value can read it (M16 and F1 are skipped).

Inputs: results/numbers/v1.1/step_{0,325}.jsonl, results/numbers/v1.2/step_449.jsonl,
        results/modal/finetuned/<set>/{final_a100,numbers_v2}/predictions.jsonl
Outputs: results/numbers/{dev_split.csv,test_drift.csv,number_accuracy.csv,run_info_eval.json}
         and results/numbers/compare/nodigit_<set>.json
  PYTHONPATH=src .venv/bin/python analysis/number_fix_eval.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.numbers import digit_value, numbers_in  # noqa: E402
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402

DIGIT = re.compile(r"[0-9০-৯]")
OUT = Path("results/numbers")
DEV = {
    "v1.0": Path("results/numbers/v1.1/step_0.jsonl"),
    "v1.1": Path("results/numbers/v1.1/step_325.jsonl"),
    "v1.2": Path("results/numbers/v1.2/step_449.jsonl"),
}
DRIFT_SETS = ["splits_fa-eval", "fleurs-test", "cv-test", "kathbath-valid"]
ACC_SETS = ["splits_fa-eval", "fleurs-test"]
ACC_RUNS = (("v1.0", "final_a100"), ("v1.2", "numbers_v2"))


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_list(text) -> list:
    return normalize_bangla_text(str(text)).split()


def has_run(words, n=5) -> bool:
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= n:
            return True
    return False


def pred_path(set_name: str, run: str) -> Path:
    return Path("results/modal/finetuned") / set_name / run / "predictions.jsonl"


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket_scores(rows) -> dict:
    edits = {"nodigit": 0, "digit": 0, "all": 0}
    refs = {"nodigit": 0, "digit": 0, "all": 0}
    counts = {"nodigit": 0, "digit": 0, "all": 0}
    loops = 0
    for row in rows:
        ref_w, hyp_w = word_list(row["ref"]), word_list(row["hyp"])
        dist = Levenshtein.distance(ref_w, hyp_w)
        key = "digit" if DIGIT.search(str(row["ref"])) else "nodigit"
        for k in (key, "all"):
            edits[k] += dist
            refs[k] += len(ref_w)
            counts[k] += 1
        loops += has_run(hyp_w)
    out = {"loops": loops}
    for k in ("nodigit", "digit", "all"):
        out[f"clips_{k}"] = counts[k]
        out[f"wer_{k}"] = 100.0 * edits[k] / refs[k] if refs[k] else float("nan")
    return out


def number_accuracy(path: Path):
    right = total = 0
    for row in load_jsonl(path):
        hyp_vals = set(numbers_in(word_list(row["hyp"])))
        for tok in word_list(row["ref"]):
            if not DIGIT.search(tok):
                continue
            value = digit_value(tok)
            if value is None:  # digit glued to a letter (M16, F1); not a number value
                continue
            total += 1
            right += value in hyp_vals
    return right, total


def dev_table() -> pd.DataFrame:
    rows = [{"model": model, **bucket_scores(load_jsonl(path))} for model, path in DEV.items()]
    return pd.DataFrame(rows)


def drift_table() -> pd.DataFrame:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    rows = []
    compare = OUT / "compare"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for set_name in DRIFT_SETS:
            base = load_jsonl(pred_path(set_name, "final_a100"))
            keep = {r["utt_id"] for r in base if not DIGIT.search(str(r["ref"]))}
            other = [r for r in load_jsonl(pred_path(set_name, "numbers_v2")) if r["utt_id"] in keep]
            a_path = tmp / f"{set_name}_v10.jsonl"
            b_path = tmp / f"{set_name}_v12.jsonl"
            write_jsonl(a_path, [r for r in base if r["utt_id"] in keep])
            write_jsonl(b_path, other)
            name = f"nodigit_{set_name}"
            subprocess.run(
                [sys.executable, "analysis/paired_compare.py",
                 "--a", str(a_path), "--b", str(b_path), "--name", name, "--out_dir", str(compare)],
                check=True, env=env,
            )
            res = json.loads((compare / f"{name}.json").read_text(encoding="utf-8"))
            lo, hi = res["diff_ci95"]
            rows.append({
                "set": set_name, "clips": int(res["n_utterances"]),
                "wer_v10": 100 * res["wer_a"], "wer_v12": 100 * res["wer_b"],
                "diff": 100 * res["diff_b_minus_a"], "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                "significant": not (lo <= 0 <= hi),
            })
    return pd.DataFrame(rows)


def accuracy_table() -> pd.DataFrame:
    rows = []
    for set_name in ACC_SETS:
        for model, run in ACC_RUNS:
            right, total = number_accuracy(pred_path(set_name, run))
            rows.append({"set": set_name, "model": model, "right": right, "total": total,
                         "share": right / total if total else float("nan")})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = list(DEV.values())
    for set_name in DRIFT_SETS:
        inputs += [pred_path(set_name, "final_a100"), pred_path(set_name, "numbers_v2")]
    with RunRecord(OUT / "run_info_eval.json", inputs=inputs) as rec:
        dev = dev_table()
        drift = drift_table()
        acc = accuracy_table()
        dev.to_csv(OUT / "dev_split.csv", index=False, encoding="utf-8-sig")
        drift.to_csv(OUT / "test_drift.csv", index=False, encoding="utf-8-sig")
        acc.to_csv(OUT / "number_accuracy.csv", index=False, encoding="utf-8-sig")
        rec.outputs = [OUT / "dev_split.csv", OUT / "test_drift.csv", OUT / "number_accuracy.csv",
                       *[OUT / "compare" / f"nodigit_{s}.json" for s in DRIFT_SETS]]
        print("\ndev_split.csv")
        print(dev.round(2).to_string(index=False))
        print("\ntest_drift.csv")
        print(drift.round(2).to_string(index=False))
        print("\nnumber_accuracy.csv")
        print(acc.assign(share=(100 * acc["share"]).round(1)).to_string(index=False))


if __name__ == "__main__":
    main()
