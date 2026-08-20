import pandas as pd
import matplotlib.pyplot as plt

# IEEE Style Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 3,
    "ps.fonttype": 3,
})

# Load the full manifest generated in Phase 1
# Update path if executing from a different root directory
df = pd.read_csv("data/processed/chunks/chunks_manifest_with_scores.csv")

# --- Figure A: Fuzzy Alignment Score Distribution ---
fig1, ax1 = plt.subplots(figsize=(3.5, 2.4))
n, bins, patches = ax1.hist(df['align_score'].dropna(), bins=40, edgecolor='black', linewidth=0.5)

# Color the dropped vs retained regions
for i in range(len(bins)-1):
    if bins[i] < 50:
        patches[i].set_facecolor('#D55E00') # Orange for dropped chunks
    else:
        patches[i].set_facecolor('#56B4E9') # Blue for retained chunks

ax1.axvline(x=50, color='black', linestyle='--', linewidth=1.2, label=r'Cutoff ($\tau=50$)')
ax1.set_xlabel('RapidFuzz Alignment Score')
ax1.set_ylabel('Chunk Count')
ax1.set_xlim(0, 100)
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.legend(loc='upper left', frameon=True)

plt.tight_layout(pad=0.5)
fig1.savefig('figures/fig_alignment_scores.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- Figure B: Chunk Duration Profile (Retained Chunks Only) ---
retained_df = df[df['align_score'] >= 50]

fig2, ax2 = plt.subplots(figsize=(3.5, 2.4))
ax2.hist(retained_df['duration'].dropna(), bins=30, color='#009E73', edgecolor='black', linewidth=0.5)

ax2.axvline(x=22.71, color='black', linestyle='--', linewidth=1.2, label='Mean: 22.71 s')
ax2.set_xlabel('Audio Chunk Duration (s)')
ax2.set_ylabel('Chunk Count')
ax2.set_xlim(0, 26)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout(pad=0.5)
fig2.savefig('figures/fig_chunk_durations.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig2)