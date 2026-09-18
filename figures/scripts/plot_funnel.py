#!/usr/bin/env python3
"""
Plots Dataset Pipeline Yield Funnel (Phase 1).
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(exist_ok=True)

stages = [
    "Raw audio\nfiles (113/121\ncompleted)",
    "Raw chunks\n(~25s, VAD-cut)",
    "Aligned &\nretained",
    "Train\nsplit",
    "Eval\nsplit",
]
counts = [121, 15312, 11129, 8858, 2271]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(
    stages[::-1],
    counts[::-1],
    color=["#8C8C8C", "#4C72B0", "#55A868", "#DD8452", "#DD8452"][::-1],
    edgecolor="black",
    linewidth=0.6,
)

for bar, count in zip(bars, counts[::-1]):
    ax.text(
        bar.get_width() + max(counts) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,}",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

ax.set_xlabel("Count", fontsize=10)
ax.set_title("Dataset Pipeline Yield — Phase 1", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xscale("log")
ax.set_xlim(1, 25000)

plt.tight_layout()
out_file = OUT_DIR / "dataset_funnel.pdf"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_file}")
