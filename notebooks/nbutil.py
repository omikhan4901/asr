"""Helpers shared by the result notebooks. They only read files under results/ and figures/: no audio, no GPU,
nothing is re-run. Numbers are percentages unless a column says otherwise."""

import inspect
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
pd.set_option("display.max_colwidth", 80)

SETS = {"splits_fa-eval": "DL Sprint eval", "fleurs-test": "FLEURS", "cv-test": "Common Voice",
        "kathbath-valid": "Kathbath"}
FINAL = "results/kaggle/finetuned/{}/soup_m1_full+m5_300+m5_300+m5_300_t4"
ZERO_SHOT = "results/modal/zeroshot/{}/tugstugi_a100"


def load(path):
    """A JSON file under the repository root."""
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metrics(folder):
    """WER and CER (%) with the 95% cluster-bootstrap CI of one system on one test set."""
    m = load(f"{folder}/metrics.json")
    lo, hi = m["wer_ci95"]
    return {"WER": round(100 * m["wer"], 2), "95% CI": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
            "CER": round(100 * m["cer"], 2), "clips": m["n_utterances"]}


def wer_table(systems, sets=SETS):
    """WER (%) of {label: folder pattern or list of patterns (first that exists)} on every test set."""
    rows = {}
    for label, patterns in systems.items():
        rows[label] = {}
        for s, name in sets.items():
            found = [ROOT / p.format(s) / "metrics.json" for p in ([patterns] if isinstance(patterns, str) else patterns)]
            found = [p for p in found if p.exists()]
            rows[label][name] = round(100 * json.loads(found[0].read_text())["wer"], 2) if found else None
    return pd.DataFrame(rows).T


def paired(name):
    """A paired comparison from results/compare (analysis/paired_compare.py): B minus A in points, 95% CI."""
    c = load(f"results/compare/{name}.json")
    lo, hi = c["diff_ci95"]
    return {"A": round(100 * c["wer_a"], 2), "B": round(100 * c["wer_b"], 2),
            "B - A": round(100 * c["diff_b_minus_a"], 2), "95% CI": f"[{100 * lo:+.2f}, {100 * hi:+.2f}]",
            "significant": not (lo <= 0 <= hi)}


def figure(name, width=850):
    display(Image(filename=str(ROOT / "figures" / "png" / f"{name}.png"), width=width))


def figure_data(name):
    """The numbers behind a figure (figures/data/<name>.csv, written by figures/scripts/<name>.py)."""
    return pd.read_csv(ROOT / "figures" / "data" / f"{name}.csv")


def source(obj):
    """Print the source code of a function or class, so the notebook shows exactly what was run."""
    print(inspect.getsource(obj))


def record(path):
    """The provenance record of a result: command, git commit, start time, runtime, library versions."""
    r = load(path)
    run = r["runs"][-1] if "runs" in r else r
    v = run.get("versions", {})
    display(Markdown(
        f"**Run record** `{path}`  \n`{run.get('command', '')}`  \ncommit `{str(run.get('git_commit', ''))[:10]}`"
        f"{' (dirty)' if run.get('git_dirty') else ''} · started {run.get('start', '')} · {run.get('runtime_s', '?')} s · "
        f"torch {v.get('torch', '?')}, transformers {v.get('transformers', '?')}, GPU {v.get('gpu', 'none')}"))
