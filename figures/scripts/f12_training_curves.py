"""
F12: training of the four M1 variants (DL Sprint train, 2 epochs, 956 steps): (a) training loss
(rolling mean over 10 logged points); (b) dev WER of the training-time checks (646 dev clips, fp16
greedy; not the reported harness numbers). LoRA ran steps 0-478 on an RTX 3060 and continued on the
A100; duplicate log rows from a cancelled job keep the last row per step.

Inputs: checkpoints/modal/<run>/{train_log,dev_log}.csv (pulled from the Modal volume)
Run:    ./Scripts/python.exe figures/scripts/f12_training_curves.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import OKABE, PAGE_W, ROOT, plt, save  # noqa: E402

RUNS = {"full": ("m1_full", OKABE["blue"]), "LoRA": ("m1_lora", OKABE["purple"]),
        "decoder-only": ("m1_decoder", OKABE["orange"]), "encoder-only": ("m1_encoder", OKABE["sky"])}
rows = []
fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE_W, 2.2))
for label, (run, color) in RUNS.items():
    tr = pd.read_csv(ROOT / "checkpoints/modal" / run / "train_log.csv").drop_duplicates("step", keep="last").sort_values("step")
    dv = pd.read_csv(ROOT / "checkpoints/modal" / run / "dev_log.csv").drop_duplicates("step", keep="last").sort_values("step")
    ax.plot(tr.step, tr.loss.rolling(10, min_periods=1).mean(), lw=1, color=color, label=label)
    bx.plot(dv.step, dv.dev_wer, "-o", ms=3, lw=1, color=color, label=label)
    rows += [{"run": run, "kind": "train_loss", "step": s, "value": v} for s, v in zip(tr.step, tr.loss)]
    rows += [{"run": run, "kind": "dev_wer", "step": s, "value": v} for s, v in zip(dv.step, dv.dev_wer)]
ax.set_xlabel("step")
ax.set_ylabel("training loss")
ax.set_title("(a) training loss", loc="left")
bx.set_xlabel("step (0 = before training; 478, 956 = epoch ends)")
bx.set_ylabel("dev WER (%)")
bx.set_xticks([0, 478, 956])
bx.set_ylim(0, 32)
bx.set_title("(b) training-time dev checks", loc="left")
bx.legend(frameon=False, loc="lower left")
save(fig, "f12_training_curves", pd.DataFrame(rows))
