#!/usr/bin/env python3
"""
Write and execute the result notebooks (notebooks/0*_*.ipynb). Each notebook only reads saved results (results/,
figures/) and runs in seconds on a CPU; outputs are saved in the files so they read on GitHub without running.

  PYTHONPATH=src .venv/bin/python notebooks/build_notebooks.py           # write + execute all
  PYTHONPATH=src .venv/bin/python notebooks/build_notebooks.py --only 03  # one notebook
"""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SETUP = "from nbutil import *\nimport pandas as pd"


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


NOTEBOOKS = {}

NOTEBOOKS["01_evaluation_and_baselines"] = [
    md("""
# 1. How we measure, and where the open models start

Every later notebook compares systems, so the scoring is fixed first and never changed afterwards: one text
normalizer, corpus word error rate (WER), and 95% confidence intervals that resample whole recordings or speakers
(clips from one recording are not independent). The media-speech test set is cut from the DL Sprint 4.0 recordings by
forced alignment and split by recording, so no recording is in both training and test data.
"""),
    code(SETUP),
    md("## The test sets\nClips, recordings or speakers (the bootstrap clusters) and reference words, read from the scored results."),
    code("""
rows = {}
for s, name in SETS.items():
    m = load(ZERO_SHOT.format(s) + "/metrics.json")
    rows[name] = {"clips": m["n_utterances"], "clusters (recordings / speakers)": m["n_groups"], "reference words": m["ref_words"]}
pd.DataFrame(rows).T
"""),
    md("## The normalizer and the WER\nThe exact code every number in this repository goes through:"),
    code("""
from bangla_asr.common.text import normalize_bangla_text
from bangla_asr.common.stats import edit_counts, corpus_rates
source(normalize_bangla_text); source(edit_counts); source(corpus_rates)
"""),
    md("## Open models without any training\nWER (%) on each test set. The tugstugi Whisper-medium model is the base for everything that follows."),
    code("""
wer_table({
    "tugstugi Whisper-medium": ZERO_SHOT,
    "Hishab conformer large *": "results/zeroshot/{}/hishab__titu_stt_bn_conformer_large",
    "BanglaASR (Whisper-small)": "results/zeroshot/{}/bangla-speech-processing__BanglaASR",
    "IndicWav2Vec": "results/zeroshot/{}/ai4bharat__indicwav2vec_v1_bengali",
    "wav2vec2-xls-r-300m-bn": "results/zeroshot/{}/arijitx__wav2vec2-xls-r-300m-bengali",
})
"""),
    md("\\* Hishab's model card lists Kathbath and Common Voice among its training data, so those two cells are not comparable."),
    code('metrics(ZERO_SHOT.format("splits_fa-eval"))'),
    code('record("results/modal/zeroshot/splits_fa-eval/tugstugi_a100/run_info.json")'),
    md("""
**Next:** the base model makes 22.7% word errors on media speech. The cheapest way to improve a recognizer without
training is to let it propose several transcripts and choose among them (rescoring); notebook 2 tests whether that works.
"""),
]

NOTEBOOKS["02_why_rescoring_plateaus"] = [
    md("""
# 2. Why rescoring stops helping

The model proposes 12 candidate transcripts per clip. Minimum Bayes risk (MBR) picks the candidate the others agree
with most; the oracle is the best candidate in hindsight. If MBR does not beat greedy decoding, we need to know why
before trying anything else.
"""),
    code(SETUP),
    md("## Choosing among 12 candidates, against greedy decoding (DL Sprint eval)\nB minus A in WER points, paired cluster bootstrap."),
    code("""
pd.DataFrame({sel: paired(f"eval_sampled_tugstugi_{sel}_vs_greedy") for sel in ("rank0", "mbr", "oracle")}).T.rename(
    index={"rank0": "first candidate", "mbr": "MBR", "oracle": "best candidate (oracle)"})
"""),
    md("MBR ties with greedy decoding although a perfect selector would gain about 4 points. The ceilings show how far selection could go:"),
    code('figure("f8_ceilings"); figure_data("f8_ceilings").head(8)'),
    code('figure("f9_oracle_vs_n", 450)'),
    md("""
## What the candidates agree on
For each reference word, count how many of the 12 candidates contain it. MBR fixes an error only when most candidates
already have the right word; most errors are in no candidate at all.
"""),
    code('figure("f2_mbr_fix_by_agreement", 560)'),
    md("A large share of the remaining errors are the same word in another spelling or dialect form, and those are exactly the errors all candidates share:"),
    code('figure("f3_variant_share_by_model", 450)'),
    code('figure("f1_example")'),
    code('record("results/compare/eval_sampled_tugstugi_mbr_vs_greedy.run_info.json")'),
    md("""
**Next:** a better voter cannot fix a spelling the whole list shares. Either the right spelling has to enter the list
(notebook 3, no retraining) or the model has to learn it (notebook 4).
"""),
]

NOTEBOOKS["03_list_correction"] = [
    md("""
# 3. Fixing the candidate lists without retraining

If the errors are shared spelling variants, adding the missing variants to the candidate list should raise the
ceiling, and a word n-gram language model built from the training text can then choose among them. All settings were
tuned on DL Sprint dev only and applied unchanged to every test set.
"""),
    code(SETUP),
    code("""
s = pd.read_csv(ROOT / "results/list_correction/summary.csv").set_index("set")
cols = {"greedy_wer": "greedy", "mbr_lm_wer": "LM-MBR on the list", "rewrite_wer": "rewrite rules",
        "mbr_lm_x_wer": "LM-MBR on the expanded list", "oracle_pool_wer": "oracle before", "oracle_x_wer": "oracle after expansion"}
(100 * s[list(cols)]).rename(columns=cols).round(2)
"""),
    code('figure("f6_list_correction")'),
    md("The settings chosen on DL Sprint dev:"),
    code('load("results/list_correction/tuned.json")'),
    code('record("results/list_correction/run_info.json")'),
    md("""
**Next:** without retraining the speech model, WER on media speech falls from 22.7% to 17.5%, and expanding the list
lowers the oracle from 18.3% to 13.7%. The stronger fix is to teach the model the corpus's spelling: notebook 4.
"""),
]

NOTEBOOKS["04_fine_tuning"] = [
    md("""
# 4. Fine-tuning: which parts of the model to train

Four ways to fine-tune the base model on 60 h of DL Sprint training clips: full fine-tuning, LoRA, decoder only and
encoder only. The diagnosis predicts that training the decoder (where spelling lives) matters most; five predictions
were written down before training.
"""),
    code(SETUP),
    code("""
M = "results/modal/finetuned/{}/"
wer_table({"zero-shot": ZERO_SHOT, "full fine-tuning (M1)": M + "m1_full", "LoRA": [M + "m1_lora", "results/kaggle/finetuned/{}/m1_lora_t4"],
           "decoder only": M + "m1_decoder", "encoder only": M + "m1_encoder"})
"""),
    md("LoRA's Common Voice and Kathbath rows were scored on a Kaggle T4; the same model gives the same WER on the T4 and the A100 within 0.01 points."),
    code('figure("f4_finetune_by_set")'),
    code('pd.DataFrame({s: paired(f"finetuning/{s}__m1_full_vs_tugstugi_a100") for s in SETS}).T'),
    code('figure("f12_training_curves")'),
    md("Fine-tuning removes the spelling errors first:"),
    code('figure("f5_gain_by_error_type", 450)'),
    md("Rescoring on top of the fine-tuned model adds little: the errors it could reach are gone."),
    code("""
st = pd.read_csv(ROOT / "results/stacking/summary.csv").set_index("set")
(100 * st[["greedy_wer", "rover_wer", "mbr_lm_x_wer", "oracle_pool_wer"]]).round(2).rename(columns={
    "greedy_wer": "M1 greedy", "rover_wer": "ROVER", "mbr_lm_x_wer": "LM-MBR expanded", "oracle_pool_wer": "oracle"})
"""),
    code('record("results/modal/finetuned/splits_fa-eval/m1_full/run_info.json")'),
    md("""
**Next:** the in-domain model is strong on DL Sprint but weaker on read speech. Notebook 5 trains on the other public
corpora too and asks how to mix them.
"""),
]

NOTEBOOKS["05_training_data_and_final_model"] = [
    md("""
# 5. Training data and the final model

Each public corpus was used alone (M1–M4) and mixed (M5, 10 h of each), then with every hour of the four corpora
(M5-full, 156 h) and with 144 h of OpenSLR 53 read speech added (M5-300, ~300 h). Same recipe every time.
"""),
    code(SETUP),
    code('figure("f13_matrix_heatmap")'),
    code("""
K = "results/kaggle/finetuned/{}/"
t = wer_table({"zero-shot": ZERO_SHOT, "M1: DL Sprint (60 h)": "results/modal/finetuned/{}/m1_full",
               "M5: 10 h x 4": K + "m5_merged_t4", "M5-full (156 h)": K + "m5_full_t4", "M5-300 (300 h)": K + "m5_300_t4",
               "final: 0.25 M1 + 0.75 M5-300": FINAL})
t["average"] = t.mean(axis=1).round(2); t
"""),
    code('figure("f16_data_scale", 450)'),
    md("""
## Choosing the final model
Weight averages of models from the same starting point cost nothing at inference. The rule, fixed before any test
score: lowest average WER over three development sets, adopted only if it beats M5-300 alone.
"""),
    code("""figure("f15_weight_averaging"); figure_data("f15_weight_averaging").query("panel == 'a'")"""),
    md("## Whole recordings\nThe 22 raw evaluation recordings, cut by voice detection only (no reference-guided cuts), texts joined."),
    code("""
rows = {}
for run, label in [("base_t4", "zero-shot"), ("m1_full_t4", "M1"), ("m5_full_t4", "M5-full"), ("m5_300_t4", "M5-300")]:
    m = load(f"results/kaggle/longform/{run}/metrics.json")
    rows[label] = {"WER": round(100 * m["wer"], 2), "95% CI": f"[{100 * m['wer_ci95'][0]:.2f}, {100 * m['wer_ci95'][1]:.2f}]",
                   "recordings": m["recordings"], "hours": m["audio_hours"], "real-time factor": m["rtf"]}
pd.DataFrame(rows).T
"""),
    code('record("results/kaggle/finetuned/splits_fa-eval/soup_m1_full+m5_300+m5_300+m5_300_t4/run_info.json")'),
    md("**Next:** how the final model compares with every other open Bangla model: notebook 6."),
]

NOTEBOOKS["06_comparison_with_other_models"] = [
    md("""
# 6. Against other open Bangla models

Every model below was scored with the same harness, greedy decoding, on a Kaggle T4. Cells where a model was trained on
that material are not compared (flags set before scoring).
"""),
    code(SETUP),
    code("""
Z = "results/kaggle/zeroshot/{}/"
wer_table({"final model": FINAL, "Hishab conformer large (CC-BY-NC)": "results/zeroshot/{}/hishab__titu_stt_bn_conformer_large",
           "IndicConformer 600M, CTC": Z + "ai4bharat__indic-conformer-600m-multilingual_ctc_t4",
           "IndicConformer 600M, RNN-T": Z + "ai4bharat__indic-conformer-600m-multilingual_t4",
           "Team Villagers (DL Sprint 4.0)": Z + "teamvillagers__whisper-bengali-ct2_t4",
           "zarifmahir21": Z + "zarifmahir21__whisper-medium-bangla_t4", "psdn-ai/bangla-asr": Z + "psdn-ai__bangla-asr_t4"})
"""),
    md("""
Not comparable: Hishab on Common Voice and Kathbath (its training data), Team Villagers and zarifmahir21 on DL Sprint
(trained on DL Sprint 4.0 data). psdn-ai/bangla-asr writes case endings as separate words, so its WER is high at a
normal character error rate. Paired differences, final model minus each system:
"""),
    code("""
names = {"hishab": "Hishab", "icctc": "IndicConformer CTC", "icrnnt": "IndicConformer RNN-T", "villagers": "Team Villagers", "zarif": "zarifmahir21"}
pd.DataFrame({f"{n} / {SETS[s]}": paired(f"final_vs_{k}_{s}") for k, n in names.items() for s in SETS
              if (ROOT / f"results/compare/final_vs_{k}_{s}.json").exists()}).T[["A", "B", "B - A", "95% CI", "significant"]]
"""),
    code('figure("f14_baselines")'),
    md("""
## Is zarifmahir21's DL Sprint score real?
It is the only model ahead on DL Sprint eval, and it was trained on DL Sprint 4.0 data. On the eval clips whose
reference the annotators corrected (notebook 7), a recognizer should be closer to the corrected text; a model that
learned these transcripts reproduces the original errors:
"""),
    code('pd.read_csv(ROOT / "results/labels/memorization_check.csv").round(4)'),
    md("It is the only system closer to the erroneous transcripts: its DL Sprint score is inflated by having seen the test material."),
    md("""
## Reported results of larger models (not re-scored)
Published Bengali numbers on the same public test sets; other normalizers and sometimes other splits, so indicative only.

| Model | Reported | Final model, same set | Source |
|---|---|---|---|
| IndicWhisper | WER Kathbath 16.6, Common Voice 24.7, FLEURS 20.9 | 10.28, 10.13, 15.60 | Vistaar, arXiv:2305.15386 |
| Omnilingual ASR 7B (Meta) | WER 17.41 on Common Voice + FLEURS combined | FLEURS 15.60, CV 10.13 | BuzzASR, arXiv:2609.09554 |
| MMS-1B (Meta) | WER 33.89, same combined set | | BuzzASR, arXiv:2609.09554 |
| SeamlessM4T v2 large | CER about 9.0 on FLEURS | CER 4.37 | Fleurs-SLU, arXiv:2501.06117 |
| Voxtral-Mini 3B Bangla fine-tune (non-commercial) | FLEURS WER 14.73 / CER 4.78 | 15.60 / CER 4.37 | model card |
"""),
    md("**Next:** the references themselves may be wrong. Notebook 7 checks them by hand."),
]

NOTEBOOKS["07_human_verification"] = [
    md("""
# 7. Checking the references by hand

Two native-speaker annotators, blind to the system, verified 605 reference transcripts, labeled word pairs (same word in
another spelling, dialect form, or a different word) and audited every difference between the zero-shot model and the
reference. A third party settled their disagreements by choosing one of the two answers.
"""),
    code(SETUP + "\ns = load('results/labels/summary.json')"),
    md("## Agreement between the annotators"),
    code("""
pd.DataFrame({
    "verified references": {"double-labeled": s["refs"]["agreement"]["double_labeled"], "agreement": f"{100 * s['refs']['agreement']['status_agreement']:.1f}% same status"},
    "word pairs": {"double-labeled": s["pairs"]["agreement"]["pairs"], "agreement": f"Cohen's kappa {s['pairs']['agreement']['fleiss_kappa']:.2f}"},
    "error audit": {"double-labeled": s["audit"]["cohen_kappa"]["units"], "agreement": f"Cohen's kappa {s['audit']['cohen_kappa']['kappa']:.2f}"},
}).T
"""),
    code("""
adj = load("results/labels/adjudication.json")
pd.Series([r["task"] for r in adj["resolutions"]]).value_counts().rename("disagreements settled")
"""),
    md("## Reference noise: share of reference words the annotators changed"),
    code("""
pd.DataFrame({src: {"clips": e["chunks"], "clips corrected %": round(100 * e["corrected_share"], 1),
                    "words changed %": round(100 * e["edit_rate"]["value"], 2),
                    "95% CI": f"[{100 * e['edit_rate']['ci95'][0]:.2f}, {100 * e['edit_rate']['ci95'][1]:.2f}]"}
              for src, e in s["refs"]["by_source"].items()}).T
"""),
    code('pd.read_csv(ROOT / "results/labels/noise_by_alignment.csv").round(4)'),
    md("## What the differences are (zero-shot model, DL Sprint eval)"),
    code("""
a = s["audit"]
pd.DataFrame({g: {k: round(100 * v["value"], 1) for k, v in a["shares_by_agreement"][g].items()}
              for g in ("selection_proof", "fixable")}).rename(columns={"selection_proof": "no candidate has the word", "fixable": "some candidate has it"})
"""),
    md("## Every model against the verified references"),
    code('figure("f17_verified_references"); figure_data("f17_verified_references").round(2)'),
    code('pd.read_csv(ROOT / "results/labels/adjudication_sensitivity.csv").round(4)'),
    md("""
**Next:** about a third of the differences are only spelling. Notebook 8 scores every model without counting them.
"""),
]

NOTEBOOKS["08_spelling_aware_wer"] = [
    md("""
# 8. Spelling-aware WER

A second metric next to the unchanged WER: edits the rules type as spelling convention or segmentation (দু পাশে /
দুপাশে), the two types the annotators confirmed as spelling only, are not counted. Same normalizer and alignment.
"""),
    code(SETUP),
    code("""
import importlib.util
spec = importlib.util.spec_from_file_location("saw", ROOT / "analysis/spelling_aware_wer.py")
saw = importlib.util.module_from_spec(spec); spec.loader.exec_module(saw)
source(saw.per_clip)
"""),
    code("""
S = pd.read_csv(ROOT / "results/spelling_aware/summary.csv")
for col in ("wer", "sa_wer"):
    t = (100 * S.pivot(index="system", columns="set", values=col)[list(SETS)]).rename(columns=SETS)
    t["average"] = t.mean(axis=1)
    display(Markdown(f"**{'standard WER' if col == 'wer' else 'spelling-aware WER'}**")); display(t.round(2))
"""),
    code('figure("f18_spelling_aware")'),
    md("Paired differences, final model minus each system (spelling-aware WER, points):"),
    code("""
P = pd.read_csv(ROOT / "results/spelling_aware/paired.csv").query("metric == 'sa_wer'")
P["text"] = [f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" for d, lo, hi in zip(P.final_minus_system, P.ci_lo, P.ci_hi)]
P.pivot(index="system", columns="set", values="text")[list(SETS)].rename(columns=SETS)
"""),
    md("""
**Conclusion:** spelling is 15–50% of every system's errors, the comparisons keep their direction, and about 40% of
the fine-tuning gain on DL Sprint is the corpus's spelling convention; the rest is a real cut in recognition errors.
"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="prefix of one notebook, e.g. 03")
    args = ap.parse_args()
    for name, cells in NOTEBOOKS.items():
        if args.only and not name.startswith(args.only):
            continue
        nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                            "language": "python"}})
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", f"notebooks/{name}.ipynb")


if __name__ == "__main__":
    main()
