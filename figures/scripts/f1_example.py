#!/usr/bin/env python3
"""
Figure 2 (example): one DL Sprint eval clip and its 12 sampled candidates. Where the candidates disagree, the majority
is right and voting fixes the word; where all 12 share a spelling variant, no candidate holds the reference word and no
rescoring can fix it. The fine-tuned model writes the reference spelling.

Inputs: data/nbest_fa/eval_sampled.jsonl (12 sampled candidates, T 0.7), results/modal/zeroshot/splits_fa-eval/
        tugstugi_a100/predictions.jsonl, results/kaggle/finetuned/splits_fa-eval/<FINAL>/predictions.jsonl
  PYTHONPATH=src ./Scripts/python.exe figures/scripts/f1_example.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from html_fig import ACCENT, GRID, INK, INK2, INK3, ROOT, VARIANT_BG, VARIANT_INK, render  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from bangla_asr.common.text import normalize_bangla_text as N  # noqa: E402

UTT = "train_033_0003"   # found by searching eval for clips where all 12 candidates share one variant (PROGRESS §69)
FINAL = "soup_m1_full+m5_300+m5_300+m5_300_t4"  # the final model (PROGRESS §70)
GLOSS = "“I was so worried about you. This time you will live.”"


def pred(path):
    for line in (ROOT / path).open(encoding="utf-8"):
        r = json.loads(line)
        if r["utt_id"] == UTT:
            return N(r["hyp"]).split()
    raise SystemExit(f"{UTT} not in {path}")


rec = next(json.loads(l) for l in (ROOT / "data/nbest_fa/eval_sampled.jsonl").open(encoding="utf-8")
           if json.loads(l)["utt_id"] == UTT)
ref = N(rec["ground_truth"]).split()
cands = [N(c["text"]).split() for c in rec["candidates"][:12]]
assert len(cands) == 12 and all(len(c) == len(ref) for c in cands), "example must align word by word"
greedy, final = pred("results/modal/zeroshot/splits_fa-eval/tugstugi_a100/predictions.jsonl"), \
    pred(f"results/kaggle/finetuned/splits_fa-eval/{FINAL}/predictions.jsonl")
differ = [sum(c[i] != ref[i] for c in cands) for i in range(len(ref))]
shared = [i for i, d in enumerate(differ) if d == 12]
voted = [i for i, d in enumerate(differ) if 0 < d < 6]
assert shared, "no word shared by all 12 candidates"


def cell(w, r, cls=""):
    return f'<td class="bn {cls} {"diff" if w != r else ""}">{w}</td>'


head = "".join(f'<th class="bn {"sh" if i in shared else ""}">{w}</th>' for i, w in enumerate(ref))
rows = "".join(f'<tr><td class="ix">{k + 1}</td>' + "".join(cell(w, ref[i], "sh" if i in shared else "")
                                                            for i, w in enumerate(c)) + "</tr>" for k, c in enumerate(cands))
counts = "".join(f'<td class="cnt {"sh" if i in shared else ""}">{12 - d}/12</td>' for i, d in enumerate(differ))
final_line = " ".join(f'<span class="{"ok" if w == r else "bad"}">{w}</span>' for w, r in zip(final, ref))
greedy_line = " ".join(f'<span class="{"" if w == r else "badg"}">{w}</span>' for w, r in zip(greedy, ref))
s = shared[0]
v = cands[0][s]
vw = " and ".join(f'<span class="bn">{ref[i]}</span>' for i in voted)

body = f'''
<div class="title">Twelve candidates, one shared spelling: the error no rescoring can fix</div>
<div class="sub">One clip from DL Sprint eval. Each row is one of the 12 candidates sampled from the zero-shot model;
shaded words differ from the reference transcript.</div>
<div class="grid">
<table>
<thead><tr><th class="ix lab">ref.</th>{head}</tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td class="ix lab">match</td>{counts}</tr></tfoot>
</table>
<div class="gloss">{GLOSS}</div>
</div>
<div class="side">
  <div class="blk"><div class="big" style="color:{VARIANT_INK}">12 of 12</div>
    <div class="txt">candidates write <span class="bn chip">{v}</span>, the reference writes
    <span class="bn chip2">{ref[s]}</span>. Rescoring can only choose among the candidates, so it can never output the
    reference word: a selection-proof error. It is the same word in another spelling.</div></div>
  <div class="blk"><div class="big" style="color:{INK}">{min(differ[i] for i in voted)} of 12</div>
    <div class="txt">candidates differ on {vw}. Here the majority is right, so voting over the list gets the
    word right: rescoring helps only where most candidates already agree with the reference.</div></div>
  <div class="blk last"><div class="k">Zero-shot, greedy</div><div class="out bn">{greedy_line}</div>
    <div class="k" style="margin-top:14px;color:{ACCENT}">Final model, fine-tuned</div><div class="out bn">{final_line}</div>
    <div class="txt" style="margin-top:8px">Fine-tuning learns the corpus's spelling, which no candidate list contains.</div></div>
</div>'''
css = f'''
.title {{ position:absolute; left:40px; top:30px; font-size:30px; font-weight:700; letter-spacing:-0.01em; }}
.sub {{ position:absolute; left:40px; top:78px; width:1320px; font-size:16px; color:{INK2}; line-height:1.45; }}
.grid {{ position:absolute; left:34px; top:140px; }}
table {{ border-collapse:separate; border-spacing:4px 3px; }}
th {{ font-size:21px; font-weight:700; padding:6px 12px 10px; text-align:center; border-bottom:2px solid {INK}; }}
td {{ font-size:18px; padding:3px 12px; text-align:center; border-radius:5px; color:{INK2}; }}
td.diff {{ background:{VARIANT_BG}; color:{VARIANT_INK}; font-weight:600; }}
th.sh {{ color:{VARIANT_INK}; }}
td.ix {{ font-size:13px; color:{INK3}; padding-right:6px; text-align:right; }}
th.ix, td.lab {{ font-size:13px; color:{INK3}; font-weight:500; border-bottom:none; text-align:right; }}
td.cnt {{ font-size:15px; font-weight:600; color:{INK}; padding-top:10px; border-top:2px solid {INK}; border-radius:0; }}
td.cnt.sh {{ color:{VARIANT_INK}; }}
.gloss {{ margin:12px 0 0 46px; font-size:15px; color:{INK2}; font-style:italic; }}
.side {{ position:absolute; left:700px; top:150px; width:640px; border-left:1px solid {GRID}; padding-left:36px; }}
.blk {{ padding-bottom:20px; margin-bottom:20px; border-bottom:1px solid {GRID}; }}
.blk.last {{ border-bottom:none; }}
.big {{ font-size:44px; font-weight:700; letter-spacing:-0.02em; line-height:1.05; }}
.txt {{ font-size:15px; color:{INK2}; line-height:1.5; margin-top:6px; }}
.chip {{ background:{VARIANT_BG}; color:{VARIANT_INK}; font-weight:600; padding:1px 6px; border-radius:4px; }}
.chip2 {{ border:1.5px solid {INK}; color:{INK}; font-weight:600; padding:0 6px; border-radius:4px; }}
.k {{ font-size:13px; font-weight:600; color:{INK3}; text-transform:uppercase; letter-spacing:0.04em; }}
.out {{ font-size:19px; margin-top:4px; color:{INK}; }}
.out .badg {{ background:{VARIANT_BG}; color:{VARIANT_INK}; font-weight:600; padding:0 4px; border-radius:4px; }}
.out .bad {{ background:{VARIANT_BG}; color:{VARIANT_INK}; }}
.out .ok {{ color:{INK}; }}
'''
data = pd.DataFrame([{"row": "reference", **{f"w{i + 1}": w for i, w in enumerate(ref)}}] +
                    [{"row": f"candidate {k + 1}", **{f"w{i + 1}": w for i, w in enumerate(c)}} for k, c in enumerate(cands)] +
                    [{"row": "zero-shot greedy", **{f"w{i + 1}": w for i, w in enumerate(greedy)}},
                     {"row": "final model", **{f"w{i + 1}": w for i, w in enumerate(final)}}])
render("f1_example", body, 1400, 720, css, data)
