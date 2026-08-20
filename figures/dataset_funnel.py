"""
Additional figures for the report, built from numbers already confirmed in
this project (dataset counts, sweep results). Run: python figures2/make_more_charts.py
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Figure 3: Dataset funnel (Phase 1 yield) ────────────────────────────────

stages = ["Raw audio\nfiles (113/121\ncompleted)", "Raw chunks\n(~25s, VAD-cut)", "Aligned &\nretained", "Train\nsplit", "Eval\nsplit"]
counts = [121, 15312, 11129, 8858, 2271] 

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(stages[::-1], counts[::-1], color=["#8C8C8C", "#4C72B0", "#55A868", "#DD8452", "#DD8452"][::-1],
                edgecolor="black", linewidth=0.6)

for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{count:,}", va="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Count", fontsize=10)
ax.set_title("Dataset Pipeline Yield — Phase 1", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xscale("log")
ax.set_xlim(1, 20000)

plt.tight_layout()
plt.savefig(OUT_DIR / "dataset_funnel..pdf", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved figures2/dataset_funnel..pdf")


# ── Figure 4: Beam diversity sweep — oracle gap by config ──────────────────

configs = ["n=5\ntemp=0.5", "n=5\ntemp=0.7", "n=10\ntemp=0.5", "n=12\ntemp=0.7"]
oracle_gaps = [3.62, 3.66, 4.10, 4.81]
one_best_wers = [31.95, 33.26, 31.90, 31.92]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

colors = ["#8C8C8C", "#8C8C8C", "#8C8C8C", "#55A868"]  # highlight chosen config
bars1 = ax1.bar(configs, oracle_gaps, color=colors, edgecolor="black", linewidth=0.6)
for bar, val in zip(bars1, oracle_gaps):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
              f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
ax1.set_ylabel("Oracle Gap (WER points)", fontsize=10)
ax1.set_title("Oracle Headroom by Beam Config\n(100-utterance calibration sweep)", fontsize=10, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, 5.5)

bars2 = ax2.bar(configs, one_best_wers, color=colors, edgecolor="black", linewidth=0.6)
for bar, val in zip(bars2, one_best_wers):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
              f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylabel("1-best WER (%)", fontsize=10)
ax2.set_title("1-best WER by Beam Config\n(quality held roughly constant)", fontsize=10, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 38)

fig.suptitle("", fontsize=1)  # spacer
plt.tight_layout()
plt.savefig(OUT_DIR / "beam_diversity_sweep..pdf", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved figures2/beam_diversity_sweep..pdf")

print("\nBoth figures saved to figures2/ at 300dpi.")
print("\nNOTE: chosen config (n=12, temp=0.7) is highlighted in green — it")
print("maximizes oracle gap without meaningfully changing 1-best WER.")