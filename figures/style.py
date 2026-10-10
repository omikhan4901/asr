"""
Shared figure style (PLAN 7b): Okabe-Ito colours, paper-sized fonts, vector PDF.

Every figure script imports this, writes the plotted numbers to
figures/data/<name>.csv and the PDF to figures/<name>.pdf (and to
paper/figures/ when the local paper folder exists).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COL_W, PAGE_W = 3.3, 6.9  # inches: one / two IEEE columns
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
         "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F"}
SET_COLOR = {"DL Sprint eval": OKABE["blue"], "FLEURS": OKABE["orange"], "Kathbath valid": OKABE["green"]}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name, data=None):
    """Write figures/<name>.pdf (+ paper/figures/) and figures/data/<name>.csv."""
    out = [ROOT / "figures" / f"{name}.pdf"]
    if (ROOT / "paper").is_dir():
        (ROOT / "paper" / "figures").mkdir(exist_ok=True)
        out.append(ROOT / "paper" / "figures" / f"{name}.pdf")
    for p in out:
        fig.savefig(p, metadata={"CreationDate": None, "ModDate": None})
    (ROOT / "figures" / "png").mkdir(exist_ok=True)  # PNG copy for the README (GitHub cannot show PDF inline)
    fig.savefig(ROOT / "figures" / "png" / f"{name}.png", dpi=200, metadata={"Software": None})
    if os.environ.get("FIG_PREVIEW"):  # optional PNG for a quick look
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / f"{name}.png", dpi=200)
    if data is not None:
        data.to_csv(ROOT / "figures" / "data" / f"{name}.csv", index=False, encoding="utf-8")
    print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in out))
