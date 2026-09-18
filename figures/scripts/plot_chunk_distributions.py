#!/usr/bin/env python3
"""
Plots Alignment Score Distribution and Dynamic Chunk Duration Profile.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# IEEE Style Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent
manifest_path = PROJECT_ROOT / "data/processed/chunks/chunks_manifest_with_scores.csv"

if not manifest_path.exists():
    print(f"Manifest not found at {manifest_path}, skipping distribution plots.")
    exit(0)

df = pd.read_csv(manifest_path)

# --- Figure A: Fuzzy Alignment Score Distribution ---
fig1, ax1 = plt.subplots(figsize=(3.5, 2.4))
n, bins, patches = ax1.hist(df["align_score"].dropna(), bins=40, edgecolor="black", linewidth=0.5)

for i in range(len(bins) - 1):
    if bins[i] < 50:
        patches[i].set_facecolor("#D55E00")
    else:
        patches[i].set_facecolor("#56B4E9")

ax1.axvline(x=50, color="black", linestyle="--", linewidth=1.2, label=r"Cutoff ($\tau=50$)")
ax1.set_xlabel("RapidFuzz Alignment Score")
ax1.set_ylabel("Chunk Count")
ax1.set_xlim(0, 100)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", frameon=True)

plt.tight_layout(pad=0.5)
f1 = OUT_DIR / "fig_alignment_scores.pdf"
fig1.savefig(f1, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {f1}")

# --- Figure B: Chunk Duration Profile (Retained Chunks Only) ---
retained_df = df[df["align_score"] >= 50]
mean_dur = retained_df["duration"].dropna().mean()

fig2, ax2 = plt.subplots(figsize=(3.5, 2.4))
ax2.hist(retained_df["duration"].dropna(), bins=30, color="#009E73", edgecolor="black", linewidth=0.5)
ax2.axvline(x=mean_dur, color="black", linestyle="--", linewidth=1.2, label=f"Mean: {mean_dur:.2f} s")
ax2.set_xlabel("Audio Chunk Duration (s)")
ax2.set_ylabel("Chunk Count")
ax2.set_xlim(0, 26)
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", frameon=True)

plt.tight_layout(pad=0.5)
f2 = OUT_DIR / "fig_chunk_durations.pdf"
fig2.savefig(f2, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {f2}")
