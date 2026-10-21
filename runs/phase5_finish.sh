#!/usr/bin/env bash
# Phase 5 finish (CPU, ~20-40 min): after the Phase 5 Kaggle sessions are unpacked under data/phase5/<any>/ (with or without the phase4/ level).
#   1. Kaggle results -> results/kaggle (never overwrites)
#   2. paired tests: M5-full vs M5, M5-300 vs M5-full, both vs zero-shot / M1 full / Hishab (FLEURS, eval)
#   3. ROVER on M5-300's own 12-best (pre-registered target, chosen before any Phase 5 score): 12_raise_ceiling.py
#   4. whole-recording WER of zero-shot, M1 full, M5-300 (results/longform)
#   5. results/phase5_summary.txt  <- the one file to read
#   bash runs/phase5_finish.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8
PY=./Scripts/python.exe
OUT=results/compare/phase5
mkdir -p results/kaggle "$OUT"

echo "== 1. Kaggle results -> results/kaggle"
for r in data/phase5/*/phase4/results data/phase5/*/results; do [ -d "$r" ] && cp -rn "$r"/* results/kaggle/; done

pair() {  # pair <a.jsonl> <b.jsonl> <name>
  if [ -f "$1" ] && [ -f "$2" ] && [ "$(wc -l < "$1")" = "$(wc -l < "$2")" ]; then
    $PY analysis/paired_compare.py --a "$1" --b "$2" --name "$3" --out_dir "$OUT" 2>&1 | tail -1
  else echo "  skip $3 (missing or incomplete)"; fi
}
echo "== 2. paired tests"
K=results/kaggle/finetuned
for s in splits_fa-dev splits_fa-eval fleurs-test cv-test kathbath-valid; do
  Z=results/modal/zeroshot/$s/tugstugi_a100/predictions.jsonl
  for m in m5_full m5_300; do pair $Z $K/$s/${m}_t4/predictions.jsonl ${s}__${m}_vs_zeroshot; done
  pair $K/$s/m5_merged_t4/predictions.jsonl $K/$s/m5_full_t4/predictions.jsonl ${s}__m5_full_vs_m5
  pair $K/$s/m5_full_t4/predictions.jsonl $K/$s/m5_300_t4/predictions.jsonl ${s}__m5_300_vs_m5_full
  pair results/modal/finetuned/$s/m1_full/predictions.jsonl $K/$s/m5_300_t4/predictions.jsonl ${s}__m5_300_vs_m1_full
done
for s in splits_fa-eval fleurs-test; do
  pair results/zeroshot/$s/hishab__titu_stt_bn_conformer_large/predictions.jsonl $K/$s/m5_300_t4/predictions.jsonl ${s}__m5_300_vs_hishab
done

echo "== 3. ROVER on M5-300's 12-best"
pools=()
for s in splits_fa-dev splits_fa-eval fleurs-test kathbath-valid; do
  p=$(ls data/phase5/*/phase4/nbest/${s}__m5_300_pool.jsonl data/phase5/*/nbest/${s}__m5_300_pool.jsonl 2>/dev/null | head -1)
  [ -n "$p" ] && pools+=("$s=$p")
done
if [ ${#pools[@]} -eq 4 ]; then
  $PY pipeline/12_raise_ceiling.py --out_dir results/phase5_stack --pools "${pools[@]}" > results/phase5_stack.log 2>&1 \
    && echo "  ROVER done (results/phase5_stack/summary.csv)" || echo "  FAILED: see results/phase5_stack.log"
else echo "  skip: ${#pools[@]} of 4 pools found"; fi

echo "== 5. summary"
$PY - <<'EOF' | tee results/phase5_summary.txt
import json, csv
from pathlib import Path
print("Phase 5 summary (WER %, B - A in points [95% cluster CI])")
for f in sorted(Path("results/compare/phase5").glob("*.json")):
    if f.name.endswith(".run_info.json"):
        continue
    d = json.loads(f.read_text(encoding="utf-8")); lo, hi = (100 * x for x in d["diff_ci95"])
    print(f"{f.stem:42s} A {100*d['wer_a']:6.2f}  B {100*d['wer_b']:6.2f}  B-A {100*d['diff_b_minus_a']:+6.2f} [{lo:+.2f}, {hi:+.2f}]")
s = Path("results/phase5_stack/summary.csv")
if s.exists():
    print("\nROVER / stacking on M5-300 (results/phase5_stack/summary.csv):")
    for row in csv.DictReader(s.open(encoding="utf-8-sig")):
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items())[:220])
print("\nWhole-recording WER (results/longform):")
for m in sorted(Path("results/kaggle/longform").glob("*/metrics.json")):
    d = json.loads(m.read_text(encoding="utf-8"))
    print(f"  {m.parent.name:20s} {100*d['wer']:6.2f} [{100*d['wer_ci95'][0]:.2f}, {100*d['wer_ci95'][1]:.2f}]  "
          f"{d['recordings']} recordings, {d['audio_hours']} h, RTF {d['rtf']}{'  (TEST MODE)' if d.get('test_mode') else ''}")
print("\nSpeed (T4 real-time factor, greedy):")
for m in sorted(Path("results/kaggle/finetuned").glob("*/m5_*_t4/metrics.json")):
    d = json.loads(m.read_text(encoding="utf-8"))
    if d.get("last_run_rtf") is not None:
        print(f"  {m.parent.parent.name + '/' + m.parent.name:45s} RTF {d['last_run_rtf']:.4f}")
EOF
echo "done: say 'phase 5 finish ran'"
