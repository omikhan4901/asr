"""
F4: WER of zero-shot tugstugi and the four M1 fine-tuning variants on every test set, all scored
with the same harness on a Modal A100, with 95% cluster-bootstrap CIs; where the A100 row is missing
(LoRA on Kathbath and Common Voice) the Kaggle T4 row of the same model is used (T4 = A100 within
0.01 WER points, results/compare/gpu_t4/).

Input: results/modal/{zeroshot,finetuned}/<set>/<run>/metrics.json, else results/kaggle/.../<run>_t4/
Run:   ./Scripts/python.exe figures/scripts/f4_finetune_by_set.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "kathbath-valid": "Kathbath valid",
        "cv-test": "Common Voice"}
RUNS = {("zeroshot", "tugstugi_a100"): ("zero-shot", OKABE["grey"]),
        ("finetuned", "m1_encoder"): ("encoder-only", OKABE["sky"]),
        ("finetuned", "m1_decoder"): ("decoder-only", OKABE["orange"]),
        ("finetuned", "m1_lora"): ("LoRA", OKABE["purple"]),
        ("finetuned", "m1_full"): ("full", OKABE["blue"])}

rows = []
for (kind, run), (label, _) in RUNS.items():
    for s, name in SETS.items():
        f = ROOT / "results/modal" / kind / s / run / "metrics.json"
        if not f.exists():
            f = ROOT / "results/kaggle" / kind / s / f"{run}_t4" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            lo, hi = m["wer_ci95"]
            rows.append({"set": name, "model": label, "wer": 100 * m["wer"], "lo": 100 * lo, "hi": 100 * hi,
                         "n": m["n_utterances"]})
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(PAGE_W, 2.3))
width = 0.16
for i, ((kind, run), (label, color)) in enumerate(RUNS.items()):
    g = d[d.model == label].set_index("set").reindex(SETS.values())
    x = [j + (i - 2) * width for j in range(len(SETS))]
    ax.bar(x, g.wer, width * 0.92, color=color, label=label)
    ax.errorbar(x, g.wer, yerr=[g.wer - g.lo, g.hi - g.wer], fmt="none", ecolor="black", lw=0.6, capsize=1.2)
ax.set_xticks(range(len(SETS)))
ax.set_xticklabels(SETS.values())
ax.set_ylabel("WER (%)")
ax.set_ylim(0, None)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
save(fig, "f4_finetune_by_set", d)
