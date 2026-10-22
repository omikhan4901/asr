"""
Figures with Bengali script (the candidate-list example, f1) as HTML/SVG, rendered to PNG and PDF by a headless Chromium (Edge or Chrome).
A browser shapes Bengali script correctly; matplotlib does not (conjuncts break).

Fonts: Inter (Latin) and Noto Sans Bengali from Google Fonts, Nirmala UI as the offline fallback.
Each figure script writes figures/data/<name>.csv, figures/html/<name>.html, figures/png/<name>.png, figures/<name>.pdf.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e0", "#ffffff"
ACCENT = "#2a78d6"   # the final model, in every designed figure
LIST = "#eb6834"     # list correction (no retraining)
BASE = "#b9b8b2"     # baselines and earlier steps
VARIANT_BG, VARIANT_INK = "#fde3d6", "#b8451a"   # a word that differs from the reference
MATCH_BG = "#eaf2fc"

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Bengali:wght@400;500;600;700&display=block" rel="stylesheet">
<style>
@page {{ size: {w}px {h}px; margin: 0; }}
html, body {{ margin: 0; padding: 0; background: {surface}; }}
body {{ width: {w}px; height: {h}px; overflow: hidden; color: {ink};
       font-family: Inter, "Noto Sans Bengali", "Nirmala UI", sans-serif; -webkit-print-color-adjust: exact; }}
.bn {{ font-family: "Noto Sans Bengali", "Nirmala UI", sans-serif; }}
{css}
</style></head><body>
"""


def browser() -> str:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe", shutil.which("chromium") or "",
              shutil.which("google-chrome") or ""):
        if p and Path(p).exists():
            return p
    raise SystemExit("no Edge/Chrome found for rendering")


def render(name: str, body: str, w: int, h: int, css: str = "", data=None):
    html = HEAD.format(w=w, h=h, surface=SURFACE, ink=INK, css=css) + body + "\n</body></html>\n"
    out_html = ROOT / "figures" / "html" / f"{name}.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    (ROOT / "figures" / "png").mkdir(exist_ok=True)
    png, pdf = ROOT / "figures" / "png" / f"{name}.png", ROOT / "figures" / f"{name}.pdf"
    common = [browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
              "--virtual-time-budget=8000", "--run-all-compositor-stages-before-draw"]
    url = out_html.as_uri()
    subprocess.run(common + [f"--window-size={w},{h}", "--force-device-scale-factor=2", f"--screenshot={png}", url],
                   check=True, capture_output=True)
    subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={pdf}", url], check=True, capture_output=True)
    if data is not None:
        data.to_csv(ROOT / "figures" / "data" / f"{name}.csv", index=False, encoding="utf-8")
    print("wrote", png.relative_to(ROOT), pdf.relative_to(ROOT), out_html.relative_to(ROOT))
