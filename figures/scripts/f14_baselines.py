#!/usr/bin/env python3
"""
Figure 14 (baselines): the final model against every open Bangla model we scored, on the four test sets, as a
leaderboard. Each cell: WER and, for the other models, the paired difference to ours (grey when the 95% interval
includes 0). Cells where a model may have trained on the test material are hatched and not ranked.

Inputs: results/kaggle/finetuned/<set>/<FINAL>/metrics.json, results/kaggle/zeroshot/<set>/<baseline>_t4/metrics.json,
        results/zeroshot/<set>/hishab__titu_stt_bn_conformer_large/, results/modal/zeroshot/<set>/tugstugi_a100/,
        results/compare/p5_final_vs_<tag>_<set>.json (analysis/paired_compare.py)
  PYTHONPATH=src ./Scripts/python.exe figures/scripts/f14_baselines.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from html_fig import ACCENT, GRID, INK, INK2, INK3, MATCH_BG, ROOT, render  # noqa: E402

FINAL = "soup_m1_full+m5_300+m5_300+m5_300_t4"
SETS = [("splits_fa-eval", "DL Sprint eval", "media speech"), ("fleurs-test", "FLEURS", "read Wikipedia"),
        ("cv-test", "Common Voice", "read, crowd"), ("kathbath-valid", "Kathbath", "read, crowd")]
IC = "results/kaggle/zeroshot/{s}/ai4bharat__indic-conformer-600m-multilingual{h}_t4"
# name, detail, {set: (result dir, compare tag)}, flags {set: mark}
ROWS = [
    ("Final model (ours)", "Whisper-medium, fine-tuned on 300 h, open",
     {s: (f"results/kaggle/finetuned/{s}/{FINAL}", None) for s, _, _ in SETS}, {}),
    ("Hishab conformer large", "CTC, ~120M, CC-BY-NC",
     {s: (f"results/zeroshot/{s}/hishab__titu_stt_bn_conformer_large", "hishab") for s, _, _ in SETS},
     {"cv-test": "‡", "kathbath-valid": "‡"}),
    ("IndicConformer 600M", "AI4Bharat, multilingual; better head per set",
     {s: None for s, _, _ in SETS}, {"kathbath-valid": "?"}),
    ("Team Villagers", "DL Sprint 4.0 finalist, Whisper (CTranslate2)",
     {s: (f"results/kaggle/zeroshot/{s}/teamvillagers__whisper-bengali-ct2_t4", "villagers") for s, _, _ in SETS},
     {"splits_fa-eval": "*"}),
    ("zarifmahir21", "Whisper-medium fine-tuned on DL Sprint-era data",
     {s: (f"results/kaggle/zeroshot/{s}/zarifmahir21__whisper-medium-bangla_t4", "zarif") for s, _, _ in SETS},
     {"splits_fa-eval": "*"}),
    ("tugstugi, zero-shot", "our starting point, Whisper-medium",
     {s: (f"results/modal/zeroshot/{s}/tugstugi_a100", "zeroshot") for s, _, _ in SETS}, {}),
]


def wer(d):
    return 100 * json.loads((ROOT / d / "metrics.json").read_text(encoding="utf-8"))["wer"]


# IndicConformer: report the better of its two heads per set (RNN-T outputs nothing on many clips over 15 s)
for s, _, _ in SETS:
    heads = [(IC.format(s=s, h=h), tag) for h, tag in (("", "icrnnt"), ("_ctc", "icctc"))]
    ROWS[2][2][s] = min(heads, key=lambda x: wer(x[0]))

cells, data = [], []
for name, detail, src, flags in ROWS:
    for s, _, _ in SETS:
        d, tag = src[s]
        v = wer(d)
        diff = ci = None
        if tag and not flags.get(s):  # flagged cells are not compared
            c = json.loads((ROOT / f"results/compare/p5_final_vs_{tag}_{s}.json").read_text(encoding="utf-8"))
            diff, ci = -100 * c["diff_b_minus_a"], [-100 * x for x in reversed(c["diff_ci95"])]  # baseline minus ours
        cells.append(dict(model=name, set=s, wer=v, diff=diff, ci=ci, flag=flags.get(s, "")))
        data.append({"model": name, "set": s, "wer": round(v, 3), "baseline_minus_final": None if diff is None else round(diff, 3),
                     "ci_low": None if ci is None else round(ci[0], 3), "ci_high": None if ci is None else round(ci[1], 3),
                     "flag": flags.get(s, ""), "source": d})
df = pd.DataFrame(cells)
best = {s: df[(df.set == s) & (df.flag == "")].wer.min() for s, _, _ in SETS}
SCALE = 32.0  # bar length: WER 0-32%


def cell(r):
    flagged = bool(r.flag)
    top = r.wer == best[r.set] and not flagged
    cls = "c" + (" best" if top else "") + (" flag" if flagged else "")
    bar = f'<div class="bar"><i style="width:{min(r.wer, SCALE) / SCALE * 100:.1f}%"></i></div>'
    sub = ""
    if r.ci is not None and not flagged:  # diff becomes NaN in the frame; ci stays None
        sig = not (r.ci[0] <= 0 <= r.ci[1])
        word = "ours better" if r.diff > 0 else "ours worse"
        sub = f'<div class="d {"sig" if sig else "ns"}">{r.diff:+.1f} {word if sig else "tie"}</div>'
    elif flagged:
        sub = '<div class="d ns">not ranked</div>'
    return f'<td class="{cls}"><div class="v">{r.wer:.1f}<span class="mk">{r.flag}</span></div>{bar}{sub}</td>'


head = "".join(f'<th><div class="sn">{n}</div><div class="sd">{d}</div></th>' for _, n, d in SETS)
rows = []
for name, detail, _, _ in ROWS:
    tds = "".join(cell(r) for r in df[df.model == name].itertuples())
    cls = ' class="ours"' if name.startswith("Final") else ""
    rows.append(f'<tr{cls}><th class="m"><div class="mn">{name}</div><div class="md">{detail}</div></th>{tds}</tr>')

body = f'''
<div class="title">Best on three of four test sets; only Hishab leads, on FLEURS</div>
<div class="sub">Word error rate (%), lower is better: every model scored greedily with the same harness and normalizer.
Blue: the lowest WER among rankable rows. Under each number: that model's WER minus ours, from a paired bootstrap
(“tie” when the 95% interval includes zero).</div>
<table>
<thead><tr><th class="m"></th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<div class="foot"><b>*</b> trained on DL Sprint 4.0 data, which includes these recordings. <b>‡</b> trained on this corpus
(model card). <b>?</b> training data not stated; AI4Bharat built Kathbath. IndicConformer: the better of its CTC and RNN-T
heads per set (its RNN-T decoder returns nothing on many clips over 15 s). Not shown: psdn-ai/bangla-asr, which writes case
endings as separate words (WER 63–76%, CER 10–17%).</div>'''
css = f'''
.title {{ position:absolute; left:40px; top:30px; font-size:30px; font-weight:700; letter-spacing:-0.01em; }}
.sub {{ position:absolute; left:40px; top:78px; width:1300px; font-size:16px; color:{INK2}; line-height:1.45; }}
table {{ position:absolute; left:34px; top:150px; border-collapse:separate; border-spacing:6px 0; }}
thead th {{ text-align:left; padding:0 14px 12px; border-bottom:2px solid {INK}; vertical-align:bottom; }}
.sn {{ font-size:17px; font-weight:700; color:{INK}; }}
.sd {{ font-size:13px; color:{INK3}; margin-top:2px; font-weight:500; }}
th.m {{ width:330px; text-align:left; padding:12px 10px 12px 6px; vertical-align:middle; border-bottom:1px solid {GRID}; }}
thead th.m {{ border-bottom:2px solid {INK}; }}
.mn {{ font-size:17px; font-weight:600; color:{INK}; }}
.md {{ font-size:13px; color:{INK3}; margin-top:2px; font-weight:400; }}
tr.ours .mn {{ color:{ACCENT}; font-weight:700; }}
td.c {{ width:225px; padding:10px 14px; border-bottom:1px solid {GRID}; vertical-align:middle; }}
.v {{ font-size:24px; font-weight:600; color:{INK}; letter-spacing:-0.01em; }}
.mk {{ font-size:15px; color:{INK2}; margin-left:3px; vertical-align:top; }}
.bar {{ height:5px; background:{GRID}; border-radius:3px; margin-top:6px; width:170px; }}
.bar i {{ display:block; height:5px; background:#b9b8b2; border-radius:3px; }}
td.best {{ background:{MATCH_BG}; border-radius:8px; }}
td.best .v {{ color:{ACCENT}; font-weight:700; }}
td.best .bar i {{ background:{ACCENT}; }}
td.flag {{ background:repeating-linear-gradient(135deg, #f3f2ee 0 6px, #ffffff 6px 12px); }}
td.flag .v {{ color:{INK3}; }}
.d {{ font-size:13px; margin-top:6px; font-weight:600; }}
.d.sig {{ color:{INK2}; }}
.d.ns {{ color:{INK3}; font-weight:500; }}
.foot {{ position:absolute; left:40px; top:690px; width:1320px; font-size:13px; color:{INK2}; line-height:1.5; }}
'''
render("f14_baselines", body, 1400, 770, css, pd.DataFrame(data))
