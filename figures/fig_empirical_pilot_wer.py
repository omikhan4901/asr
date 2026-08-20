"""
Generates the two key figures for the report:
  1. wer_comparison.pdf — bar chart of all WER numbers (1-best, MBR, oracle,
     cited LoRA, cited full fine-tune) on the eval split.
  2. oracle_gap_recovery.pdf — shows how much of the oracle headroom MBR
     recovers, on both splits, side by side.

Run: python figures/make_charts.py
Outputs land in figures/ as 300dpi PNGs, ready to embed in the IEEE docx or
LaTeX draft.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # safe default; swap for
                                                       # Times New Roman if
                                                       # your LaTeX/Word env
                                                       # has it installed
from pathlib import Path

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Figure 1: WER comparison bar chart (eval split, your headline table) ───

methods = [
    "1-best\n(baseline)",
    "MBR\n(ours, inference-only)",
    "Oracle\n(ceiling, unreachable)",
    "LoRA fine-tune\n(cited, Bhuiyan et al.)",
    "Full fine-tune\n(cited, Bhuiyan et al.)",
]
wers = [34.85, 32.29, 28.72, 31.32, 24.41]
# color: your own results in one color family, cited/reference results in another
colors = ["#4C72B0", "#55A868", "#8C8C8C", "#C44E52", "#C44E52"]
is_cited = [False, False, False, True, True]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(methods, wers, color=colors, edgecolor="black", linewidth=0.6)

for bar, wer, cited in zip(bars, wers, is_cited):
    label = f"{wer:.2f}%" + (" *" if cited else "")
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             label, ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Word Error Rate (%)", fontsize=10)
ax.set_ylim(0, 40)
ax.set_title("WER Comparison — Eval Split (N=2,271)", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=8.5, rotation=15, ha="right")
plt.subplots_adjust(bottom=0.28)
ax.text(0.5, -0.32, "* cited from prior work (Bhuiyan et al., arXiv:2605.08214), not measured by this project",
        transform=ax.transAxes, ha="center", fontsize=7.5, style="italic", color="dimgray")

plt.savefig(OUT_DIR / "wer_comparison.pdf", dpi=300, bbox_inches="tight")
# ── Figure 2: Oracle gap recovery, side-by-side eval vs train ──────────────

splits = ["Eval\n(N=2,271)", "Train\n(N=8,858)"]
one_best = [34.85, 36.62]
mbr = [32.29, 33.66]
oracle = [28.72, 29.83]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = range(len(splits))
width = 0.25

bars1 = ax.bar([i - width for i in x], one_best, width, label="1-best baseline", color="#4C72B0")
bars2 = ax.bar([i for i in x], mbr, width, label="MBR (ours)", color="#55A868")
bars3 = ax.bar([i + width for i in x], oracle, width, label="Oracle ceiling", color="#8C8C8C")

for bars in (bars1, bars2, bars3):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

# annotate recovery percentage above each split's group
recovery_pct = ["~41.8%", "~43.7%"]
for i, pct in enumerate(recovery_pct):
    ax.text(i, 38.5, f"MBR recovers {pct}\nof oracle gap",
             ha="center", fontsize=8, fontweight="bold", color="#2b5c34")

ax.set_xticks(list(x))
ax.set_xticklabels(splits, fontsize=9)
ax.set_ylabel("Word Error Rate (%)", fontsize=10)
ax.set_ylim(0, 42)
ax.set_title("MBR Rescoring: Oracle Gap Recovery Across Splits", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "oracle_gap_recovery.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved figures/oracle_gap_recovery.pdf")

print("\nBoth figures saved to figures/ at 300dpi — ready to embed in the report.")