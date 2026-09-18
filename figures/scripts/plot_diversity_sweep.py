#!/usr/bin/env python3
"""
Plots Beam Diversity Sweep: Oracle Gap & 1-Best WER across Configurations.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(exist_ok=True)

configs = ["n=5\ntemp=0.5", "n=5\ntemp=0.7", "n=10\ntemp=0.5", "n=12\ntemp=0.7"]
oracle_gaps = [3.62, 3.66, 4.10, 4.81]
one_best_wers = [31.95, 33.26, 31.90, 31.92]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

colors = ["#8C8C8C", "#8C8C8C", "#8C8C8C", "#55A868"]  # Highlight chosen config
bars1 = ax1.bar(configs, oracle_gaps, color=colors, edgecolor="black", linewidth=0.6)
for bar, val in zip(bars1, oracle_gaps):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f"{val:.2f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax1.set_ylabel("Oracle Gap (WER points)", fontsize=10)
ax1.set_title("Oracle Headroom by Beam Config\n(100-utterance calibration sweep)", fontsize=10, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, 5.8)

bars2 = ax2.bar(configs, one_best_wers, color=colors, edgecolor="black", linewidth=0.6)
for bar, val in zip(bars2, one_best_wers):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"{val:.2f}%",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax2.set_ylabel("1-best WER (%)", fontsize=10)
ax2.set_title("1-best WER by Beam Config\n(Quality held roughly constant)", fontsize=10, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 40)

plt.tight_layout()
out_file = OUT_DIR / "beam_diversity_sweep.pdf"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_file}")
