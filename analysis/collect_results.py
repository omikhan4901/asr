"""Collect every results/**/metrics.json and agreement summary into results/summary.md (+ .csv)."""
import json
from pathlib import Path

import pandas as pd

rows = []
for kind in ("zeroshot", "finetuned"):
    for f in Path("results", kind).glob("*/*/metrics.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        lo, hi = m.get("wer_ci95", [float("nan")] * 2)
        model = m["model"] if kind == "zeroshot" else f"{f.parent.name} (fine-tuned)"
        rows.append({"test_set": m["set"], "model": model, "WER%": round(100 * m["wer"], 2),
                     "WER 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "CER%": round(100 * m["cer"], 2),
                     "utts": m["n_utterances"], "groups": m.get("n_groups"), "RTF": m.get("last_run_rtf")})
out = ["# Results summary", ""]
if rows:
    df = pd.DataFrame(rows).sort_values(["test_set", "WER%"])
    df.to_csv("results/summary.csv", index=False, encoding="utf-8-sig")
    out += ["## Zero-shot and fine-tuned (greedy, current normalizer)", "", df.to_markdown(index=False), ""]
for f in Path("results/agreement").glob("*/summary.json"):
    s = json.loads(f.read_text(encoding="utf-8"))
    out += [f"## Plateau anatomy: {f.parent.name}", ""]
    out += [f"- {k}: {round(v, 4) if isinstance(v, float) else v}" for k, v in s.items()]
    hist = f.parent / "histogram.csv"
    if hist.exists():
        out += ["", pd.read_csv(hist).to_markdown(index=False)]
    out.append("")
Path("results").mkdir(exist_ok=True)
Path("results/summary.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
