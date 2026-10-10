"""
F7: main results. WER with 95% cluster-bootstrap CIs for every system on every test set, one panel per
test set. Zero-shot rows and Phase 3 were scored on the RTX 3060; fine-tuned rows on the Modal A100
(the same model gives WER within 0.02 points on both GPUs, R40). Rows whose model declared the test set
as training data are drawn hollow (contaminated, never compared). Generic Whisper large-v3 / turbo
(80-93% WER) are left out of the plot range.

Inputs: results/zeroshot/<set>/<model>/metrics.json, results/modal/finetuned/<set>/<run>/metrics.json,
        results/phase3/summary.csv
Run:    ./Scripts/python.exe figures/scripts/f7_main_results_forest.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "kathbath-valid": "Kathbath valid",
        "cv-test": "Common Voice"}
ZS = {"wav2vec2-xls-r-300m-bn": "arijitx__wav2vec2-xls-r-300m-bengali",
      "BanglaASR (Whisper-small)": "bangla-speech-processing__BanglaASR",
      "IndicWav2Vec": "ai4bharat__indicwav2vec_v1_bengali",
      "Hishab conformer large": "hishab__titu_stt_bn_conformer_large",
      "tugstugi zero-shot": "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium"}
FT = {"M1 encoder-only": "m1_encoder", "M1 decoder-only": "m1_decoder", "M1 LoRA": "m1_lora", "M1 full": "m1_full"}
CONTAMINATED = {("kathbath-valid", "Hishab conformer large"), ("cv-test", "Hishab conformer large"),
                ("cv-test", "BanglaASR (Whisper-small)")}
COLOR = {"zero-shot": OKABE["grey"], "phase3": OKABE["orange"], "finetuned": OKABE["blue"]}

rows = []
for s, sname in SETS.items():
    for label, folder in ZS.items():
        f = ROOT / "results/zeroshot" / s / folder / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"set": sname, "system": label, "kind": "zero-shot", "wer": 100 * m["wer"],
                         "lo": 100 * m["wer_ci95"][0], "hi": 100 * m["wer_ci95"][1],
                         "contaminated": (s, label) in CONTAMINATED})
    for label, run in FT.items():
        f = ROOT / "results/modal/finetuned" / s / run / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"set": sname, "system": label, "kind": "finetuned", "wer": 100 * m["wer"],
                         "lo": 100 * m["wer_ci95"][0], "hi": 100 * m["wer_ci95"][1], "contaminated": False})
p3 = pd.read_csv(ROOT / "results/phase3/summary.csv").set_index("set")
for s, sname in SETS.items():
    if s in p3.index:
        r = p3.loc[s]
        rows.append({"set": sname, "system": "Phase 3 (no training)", "kind": "phase3", "wer": 100 * r.mbr_lm_x_wer,
                     "lo": 100 * (r.greedy_wer + r.mbr_lm_x_ci_lo), "hi": 100 * (r.greedy_wer + r.mbr_lm_x_ci_hi),
                     "contaminated": False})
d = pd.DataFrame(rows)
order = list(ZS) + ["Phase 3 (no training)"] + list(FT)

fig, axes = plt.subplots(1, len(SETS), figsize=(PAGE_W, 2.9), sharey=True)
for ax, sname in zip(axes, SETS.values()):
    g = d[d.set == sname].set_index("system")
    for yi, system in enumerate(order[::-1]):
        if system not in g.index:
            continue
        r = g.loc[system]
        c = COLOR[r.kind]
        ax.errorbar(r.wer, yi, xerr=[[r.wer - r.lo], [r.hi - r.wer]], fmt="o", ms=3.5, lw=0.8, capsize=1.5,
                    color=c, mfc="white" if r.contaminated else c)
    ax.set_title(sname)
    ax.set_xlim(0, 70 if sname == "DL Sprint eval" else 50)
    ax.set_xlabel("WER (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order[::-1])
fig.text(0.99, -0.07, "hollow marker = this test set is in the model's declared training data (not compared)", ha="right", fontsize=6.5)
save(fig, "f7_main_results_forest", d)
