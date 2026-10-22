#!/usr/bin/env python3
"""
Figure 1 (hero): the paper in one picture. Left: DL Sprint eval WER from zero-shot greedy through rescoring (no gain),
the selection ceiling, list correction (no retraining) to the final fine-tuned model. Right: the average over the four
test sets, zero-shot vs the final model.

Inputs: figures/data/f8_ceilings.csv (run f8_ceilings.py first), results/modal/zeroshot/<set>/tugstugi_a100/metrics.json,
        results/kaggle/finetuned/<set>/<FINAL>/metrics.json
  PYTHONPATH=src ./Scripts/python.exe figures/scripts/f0_hero.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from html_fig import ACCENT, BASE, GRID, INK, INK2, INK3, LIST, ROOT, render  # noqa: E402

FINAL = "soup_m1_full+m5_300+m5_300+m5_300_t4"  # the final model: 0.25 M1 + 0.75 M5-300, chosen by the dev rule (PROGRESS §70)
SETS = [("splits_fa-eval", "DL Sprint eval"), ("fleurs-test", "FLEURS"), ("cv-test", "Common Voice"),
        ("kathbath-valid", "Kathbath")]


def wer(p):
    return 100 * json.loads((ROOT / p).read_text(encoding="utf-8"))["wer"]


c = pd.read_csv(ROOT / "figures/data/f8_ceilings.csv").set_index("item")["value"]
final_eval = wer(f"results/kaggle/finetuned/splits_fa-eval/{FINAL}/metrics.json")
steps = [("Zero-shot", "greedy decoding", c["greedy decoding"], "base"),
         ("Rescoring", "MBR, 12 candidates", c["MBR (12 sampled candidates)"], "base"),
         ("Ceiling", "best of 12 candidates", c["best single candidate (oracle)"], "ceiling"),
         ("List correction", "no retraining", c["List correction: LM-MBR on expanded list"], "list"),
         ("Final model", "fine-tuned, 300 h", final_eval, "final")]
per_set = [(name, wer(f"results/modal/zeroshot/{s}/tugstugi_a100/metrics.json"),
            wer(f"results/kaggle/finetuned/{s}/{FINAL}/metrics.json")) for s, name in SETS]
avg0, avg1 = (sum(r[i] for r in per_set) / len(per_set) for i in (1, 2))

# ---- left: the staircase (SVG units = CSS px)
W, H = 1400, 640
x0, base_y, scale, bw, gap = 70, 548, 14.0, 118, 46
svg = [f'<svg width="860" height="{H}" viewBox="0 0 860 {H}" style="position:absolute;left:0;top:0">']
for v in (0, 5, 10, 15, 20, 25):
    y = base_y - v * scale
    svg.append(f'<line x1="{x0 - 10}" x2="{x0 + 5 * (bw + gap) - gap + 10}" y1="{y}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
    svg.append(f'<text x="{x0 - 18}" y="{y + 5}" text-anchor="end" font-size="14" fill="{INK3}">{v}</text>')
svg.append(f'<text x="{x0 - 18}" y="{base_y - 25 * scale - 22}" text-anchor="end" font-size="13" fill="{INK3}">WER %</text>')
tops = []
for i, (lab, sub, v, kind) in enumerate(steps):
    x, h = x0 + i * (bw + gap), v * scale
    y = base_y - h
    tops.append((x, y, v))
    fill = {"base": BASE, "list": LIST, "final": ACCENT}.get(kind)
    if kind == "ceiling":
        svg.append(f'<rect x="{x + 1}" y="{y}" width="{bw - 2}" height="{h}" rx="5" fill="none" stroke="{INK3}" '
                   f'stroke-width="2" stroke-dasharray="6 5"/>')
    else:
        svg.append(f'<path d="M{x},{base_y} V{y + 6} Q{x},{y} {x + 6},{y} H{x + bw - 6} Q{x + bw},{y} {x + bw},{y + 6} '
                   f'V{base_y} Z" fill="{fill}"/>')
    weight, size, col = (700, 34, ACCENT) if kind == "final" else (600, 28, INK if kind != "ceiling" else INK2)
    svg.append(f'<text x="{x + bw / 2}" y="{y - 14}" text-anchor="middle" font-size="{size}" font-weight="{weight}" '
               f'fill="{col}">{v:.1f}</text>')
    svg.append(f'<text x="{x + bw / 2}" y="{base_y + 30}" text-anchor="middle" font-size="17" font-weight="600" '
               f'fill="{INK}">{lab}</text>')
    svg.append(f'<text x="{x + bw / 2}" y="{base_y + 52}" text-anchor="middle" font-size="14" fill="{INK2}">{sub}</text>')
svg.append(f'<line x1="{x0 - 10}" x2="{x0 + 5 * (bw + gap) - gap + 10}" y1="{base_y}" y2="{base_y}" stroke="{INK3}" stroke-width="1.2"/>')
# the plateau: bracket over zero-shot and rescoring
(xa, ya, va), (xb, yb, vb) = tops[0], tops[1]
yb_top = min(ya, yb) - 62
svg.append(f'<path d="M{xa + 10},{yb_top + 10} V{yb_top} H{xb + bw - 10} V{yb_top + 10}" fill="none" stroke="{INK2}" stroke-width="1.5"/>')
svg.append(f'<text x="{(xa + xb + bw) / 2}" y="{yb_top - 12}" text-anchor="middle" font-size="16" fill="{INK}">'
           f'<tspan font-weight="700">{vb - va:+.1f}</tspan> rescoring does not help</text>')
# drops: list correction and final model relative to greedy
for i in (3, 4):
    x, y, v = tops[i]
    d = v - steps[0][2]
    col = LIST if i == 3 else ACCENT
    svg.append(f'<text x="{x + bw / 2}" y="{y + 30}" text-anchor="middle" font-size="15" font-weight="600" fill="#ffffff">'
               f'{d:+.1f}</text>')
x, y, v = tops[2]
svg.append(f'<text x="{x + bw / 2}" y="{y + 28}" text-anchor="middle" font-size="13" fill="{INK2}">best any</text>')
svg.append(f'<text x="{x + bw / 2}" y="{y + 45}" text-anchor="middle" font-size="13" fill="{INK2}">selector can do</text>')
svg.append("</svg>")

# ---- right: the average over four test sets
rows = []
lo, hi, tw = 0, 25, 200
for name, a, b in per_set:
    pa, pb = a / hi * tw, b / hi * tw
    rows.append(f'''<div class="row"><div class="rl">{name}</div>
<svg width="{tw + 12}" height="22" style="overflow:visible"><line x1="6" x2="{tw + 6}" y1="11" y2="11" stroke="{GRID}" stroke-width="2"/>
<line x1="{pb + 6}" x2="{pa + 6}" y1="11" y2="11" stroke="{BASE}" stroke-width="3"/>
<circle cx="{pa + 6}" cy="11" r="6.5" fill="{BASE}" stroke="#fff" stroke-width="2"/>
<circle cx="{pb + 6}" cy="11" r="7.5" fill="{ACCENT}" stroke="#fff" stroke-width="2"/></svg>
<div class="rv"><span class="old">{a:.1f}</span> → <b>{b:.1f}</b></div></div>''')

body = f'''
<div class="title">Rescoring plateaus. Fixing the cause halves the error.</div>
<div class="sub">Word error rate (%), lower is better. Left: DL Sprint eval, 4,234 clips of Bangla drama, narration and news.
Right: four test sets.</div>
<div class="left">{"".join(svg)}</div>
<div class="right">
  <div class="k">Average over four test sets</div>
  <div class="hero">{avg1:.1f}<span class="pct">%</span></div>
  <div class="from">from <b>{avg0:.1f}%</b> zero-shot, one Whisper-medium at the same speed</div>
  <div class="rows">{"".join(rows)}</div>
  <div class="legend"><span class="dot" style="background:{BASE}"></span>zero-shot tugstugi
     <span class="dot" style="background:{ACCENT};margin-left:18px"></span>final model</div>
</div>'''
css = f'''
.title {{ position:absolute; left:40px; top:30px; font-size:30px; font-weight:700; letter-spacing:-0.01em; }}
.sub {{ position:absolute; left:40px; top:76px; width:1300px; font-size:16px; color:{INK2}; line-height:1.45; }}
.left {{ position:absolute; left:0; top:0; width:860px; height:{H}px; }}
.right {{ position:absolute; left:900px; top:150px; width:460px; height:440px; border-left:1px solid {GRID}; padding-left:40px; }}
.k {{ font-size:16px; font-weight:600; color:{INK2}; text-transform:none; }}
.hero {{ font-size:96px; font-weight:700; color:{ACCENT}; line-height:1.0; margin-top:10px; letter-spacing:-0.03em; }}
.pct {{ font-size:48px; margin-left:4px; }}
.from {{ font-size:15px; color:{INK2}; margin-top:8px; width:400px; line-height:1.4; }}
.rows {{ margin-top:30px; }}
.row {{ display:flex; align-items:center; height:40px; }}
.rl {{ width:130px; font-size:15px; color:{INK}; white-space:nowrap; }}
.rv {{ margin-left:14px; font-size:15px; color:{INK}; white-space:nowrap; }}
.rv .old {{ color:{INK3}; }}
.legend {{ margin-top:18px; font-size:14px; color:{INK2}; display:flex; align-items:center; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:7px; }}
'''
data = pd.DataFrame([{"panel": "DL Sprint eval", "item": f"{a} ({b})", "wer": round(v, 3)} for a, b, v, _ in steps] +
                    [{"panel": "four test sets", "item": f"{n}: zero-shot", "wer": round(a, 3)} for n, a, _ in per_set] +
                    [{"panel": "four test sets", "item": f"{n}: final", "wer": round(b, 3)} for n, _, b in per_set] +
                    [{"panel": "four test sets", "item": "average: zero-shot", "wer": round(avg0, 3)},
                     {"panel": "four test sets", "item": "average: final", "wer": round(avg1, 3)}])
render("f0_hero", body, W, H, css, data)
