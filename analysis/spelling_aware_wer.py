#!/usr/bin/env python3
"""
Spelling-aware WER: a second metric next to the (unchanged) WER. Every edit of the minimum-edit word alignment is
typed by the rules of bangla_asr.common.variants (§3.5); the edits that the human labels confirmed as spelling only
(PROGRESS R68: 'convention' 98-99% spelling, 'segmentation' 83%) are not counted as errors. A second column also
drops 'number' edits (digits vs number words; all transcript or format differences in the audit).

  SA-WER     = (edits - convention - segmentation) / reference words
  SA-WER+num = (edits - convention - segmentation - number) / reference words

Same normalizer, same alignment count as the WER numerator (asserted per clip), 95% cluster-bootstrap CIs over
recordings / speakers, and paired differences of the final model against every other system.

Inputs: each system's predictions.jsonl (results/{modal,kaggle,zeroshot}/<set>/<system>/)
Output: results/spelling_aware/summary.csv, results/spelling_aware/paired.csv
  PYTHONPATH=src .venv/bin/python analysis/spelling_aware_wer.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from bangla_asr.common.text import normalize_bangla_text  # noqa: E402
from bangla_asr.common.variants import classify_alignment  # noqa: E402

SETS = ["splits_fa-eval", "fleurs-test", "cv-test", "kathbath-valid"]
FINAL = "Final model"
SYSTEMS = {  # label -> folder pattern ({} = test set)
    "tugstugi zero-shot": "results/modal/zeroshot/{}/tugstugi_a100",
    "Hishab conformer large": "results/zeroshot/{}/hishab__titu_stt_bn_conformer_large",
    "IndicConformer CTC": "results/kaggle/zeroshot/{}/ai4bharat__indic-conformer-600m-multilingual_ctc_t4",
    "IndicConformer RNN-T": "results/kaggle/zeroshot/{}/ai4bharat__indic-conformer-600m-multilingual_t4",
    "Team Villagers": "results/kaggle/zeroshot/{}/teamvillagers__whisper-bengali-ct2_t4",
    "zarifmahir21": "results/kaggle/zeroshot/{}/zarifmahir21__whisper-medium-bangla_t4",
    "M1 (DL Sprint, 60 h)": "results/modal/finetuned/{}/m1_full",
    "M5 (10 h x 4)": "results/kaggle/finetuned/{}/m5_merged_t4",
    "M5-full (156 h)": "results/kaggle/finetuned/{}/m5_full_t4",
    "M5-300 (300 h)": "results/kaggle/finetuned/{}/m5_300_t4",
    FINAL: "results/kaggle/finetuned/{}/soup_m1_full+m5_300+m5_300+m5_300_t4",
}
SPELLING = {"convention", "segmentation"}
N_BOOT = 1000


def per_clip(path: str) -> pd.DataFrame:
    rows = []
    for line in open(Path(path) / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ref, hyp = normalize_bangla_text(r["ref"]).split(), normalize_bangla_text(r["hyp"]).split()
        edits = classify_alignment(ref, hyp)
        assert len(edits) == Levenshtein.distance(ref, hyp), r["utt_id"]
        types = [e["type"] for e in edits]
        spell = sum(t in SPELLING for t in types)
        rows.append({"utt_id": r["utt_id"], "group": str(r["group"]), "words": len(ref), "edits": len(edits),
                     "sa_edits": len(edits) - spell, "sa_num_edits": len(edits) - spell - types.count("number")})
    return pd.DataFrame(rows)


def boot_index(groups, seed=0):
    uniq = {g: k for k, g in enumerate(dict.fromkeys(groups))}
    gi = np.array([uniq[g] for g in groups])
    return gi, len(uniq), np.random.default_rng(seed).integers(0, len(uniq), size=(N_BOOT, len(uniq)))


def ratio_ci(d, col, gi, G, samp):
    num, den = np.bincount(gi, d[col].to_numpy(float), G), np.bincount(gi, d["words"].to_numpy(float), G)
    r = num[samp].sum(1) / den[samp].sum(1)
    return d[col].sum() / d["words"].sum(), np.percentile(r, [2.5, 97.5])


def main():
    jobs = [(s, label, pat.format(s)) for s in SETS for label, pat in SYSTEMS.items()]
    with ProcessPoolExecutor() as ex:
        clips = dict(zip([(s, l) for s, l, _ in jobs], ex.map(per_clip, [p for _, _, p in jobs])))
    summary, paired = [], []
    for s in SETS:
        base = clips[(s, FINAL)].set_index("utt_id")
        for label in SYSTEMS:
            d = clips[(s, label)]
            gi, G, samp = boot_index(d["group"].tolist())
            row = {"set": s, "system": label, "clips": len(d), "words": int(d["words"].sum())}
            for col, name in (("edits", "wer"), ("sa_edits", "sa_wer"), ("sa_num_edits", "sa_wer_num")):
                v, ci = ratio_ci(d, col, gi, G, samp)
                row.update({name: v, f"{name}_lo": ci[0], f"{name}_hi": ci[1]})
            row["spelling_share_of_errors"] = 1 - d["sa_edits"].sum() / d["edits"].sum()
            summary.append(row)
            if label == FINAL:
                continue
            o = d.set_index("utt_id").loc[base.index]  # same clips, same order
            gi, G, samp = boot_index(base["group"].tolist())
            for col, name in (("edits", "wer"), ("sa_edits", "sa_wer")):
                w = np.bincount(gi, base["words"].to_numpy(float), G)
                f, b = np.bincount(gi, base[col].to_numpy(float), G), np.bincount(gi, o[col].to_numpy(float), G)
                diff = f[samp].sum(1) / w[samp].sum(1) - b[samp].sum(1) / w[samp].sum(1)
                paired.append({"set": s, "system": label, "metric": name,
                               "final_minus_system": (base[col].sum() - o[col].sum()) / base["words"].sum(),
                               "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5)})
    out = Path("results/spelling_aware")
    out.mkdir(parents=True, exist_ok=True)
    S, P = pd.DataFrame(summary), pd.DataFrame(paired)
    S.to_csv(out / "summary.csv", index=False)
    P.to_csv(out / "paired.csv", index=False)
    show = S.pivot(index="system", columns="set", values="sa_wer")[SETS].loc[list(SYSTEMS)] * 100
    show["average"] = show.mean(1)
    print("SA-WER %\n", show.round(2).to_string())
    show = S.pivot(index="system", columns="set", values="wer")[SETS].loc[list(SYSTEMS)] * 100
    show["average"] = show.mean(1)
    print("WER %\n", show.round(2).to_string())


if __name__ == "__main__":
    main()
