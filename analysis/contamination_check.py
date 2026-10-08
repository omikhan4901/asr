"""
Memorization check for zero-shot rows: does a model look like it was trained on a test set?

A model that saw a test set during training reproduces many references exactly, so its
share of exact-match utterances (0 word edits after the frozen normalizer) jumps far above
other models of similar corpus WER. Compares every model in results/zeroshot/<set>/ and,
for the DL Sprint eval split, lists per-recording WER/CER so a recording seen in training
would stand out (near-zero WER on that recording only).

Outputs in results/contamination/:
  per_set.csv                  set, model, corpus WER, exact-match share [95% cluster-bootstrap CI], median utterance WER
  per_recording_splits_fa-eval.csv   recording x model WER/CER and exact-match share

  PYTHONPATH=src ./Scripts/python.exe analysis/contamination_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from bangla_asr.common.provenance import RunRecord  # noqa: E402
from bangla_asr.common.safe_io import read_jsonl  # noqa: E402
from bangla_asr.common.stats import bootstrap_ratio, edit_counts  # noqa: E402

ZS = Path("results/zeroshot")
OUT = Path("results/contamination")
SETS = ["splits_fa-eval", "fleurs-test", "kathbath-valid", "cv-test"]
FA_SET = "splits_fa-eval"


def utterance_rows(pred_path):
    rows = []
    for r in read_jsonl(pred_path):
        c = edit_counts(r["hyp"], r["ref"])
        rows.append(dict(c, utt_id=r["utt_id"], group=r["group"], exact=int(c["word_edits"] == 0),
                         utt_wer=c["word_edits"] / max(1, c["ref_words"])))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = sorted(p for s in SETS for p in (ZS / s).glob("*/predictions.jsonl"))
    with RunRecord(OUT / "run_info.json", inputs=preds) as rec:
        per_set, per_rec = [], []
        for p in preds:
            s, model = p.parent.parent.name, p.parent.name
            df = utterance_rows(p)
            ex = bootstrap_ratio(df["group"].tolist(), df["exact"], np.ones(len(df)))
            per_set.append({"set": s, "model": model, "n_utt": len(df), "n_groups": ex["n_groups"],
                            "wer": df.word_edits.sum() / df.ref_words.sum(),
                            "exact_share": ex["value"], "exact_lo": ex["ci95"][0], "exact_hi": ex["ci95"][1],
                            "median_utt_wer": df.utt_wer.median()})
            if s == FA_SET:
                g = df.groupby("group").agg(word_edits=("word_edits", "sum"), ref_words=("ref_words", "sum"),
                                            char_edits=("char_edits", "sum"), ref_chars=("ref_chars", "sum"),
                                            exact_share=("exact", "mean"), n_utt=("exact", "size")).reset_index()
                g["wer"] = g.word_edits / g.ref_words
                g["cer"] = g.char_edits / g.ref_chars
                g["model"] = model
                per_rec.append(g.rename(columns={"group": "recording"}))

        ps = pd.DataFrame(per_set).sort_values(["set", "wer"])
        ps.to_csv(OUT / "per_set.csv", index=False, float_format="%.4f")
        pr = pd.concat(per_rec)[["recording", "model", "n_utt", "wer", "cer", "exact_share"]]
        pr.to_csv(OUT / f"per_recording_{FA_SET}.csv", index=False, float_format="%.4f")
        rec.outputs = [OUT / "per_set.csv", OUT / f"per_recording_{FA_SET}.csv"]

    pd.set_option("display.width", 200)
    print(ps.assign(model=ps.model.str[:40]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    wide = pr.pivot(index="recording", columns="model", values="wer")
    print(f"\n{FA_SET}: min per-recording WER per model\n{wide.min().round(3).to_string()}")


if __name__ == "__main__":
    main()
