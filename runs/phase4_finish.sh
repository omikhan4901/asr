#!/usr/bin/env bash
# Phase 4 finish (CPU only, ~10-20 min): after the last Kaggle session (score:m1_full_4ep + score:m6_harmonized)
# is unpacked somewhere under data/phase4/, this
#   1. copies every Kaggle result folder under data/phase4/*/phase4/results into results/kaggle (never overwrites)
#   2. re-runs runs/phase4_analysis.sh (paired tests vs zero-shot, error types) for all rows
#   3. paired tests: M6 vs M5 (Gate 4), 4-epoch vs M1 full, M1 full and M1 full + ROVER vs Phase 3 (17.46)
#   4. regenerates Figure 8 and Figure 13 (M6 row appears once scored)
#   5. writes results/phase4_final_summary.txt (the file to send back / read)
#   bash runs/phase4_finish.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8
PY=./Scripts/python.exe
OUT=results/compare/phase4_final
mkdir -p results/kaggle "$OUT"

echo "== 1. Kaggle results -> results/kaggle"
for r in data/phase4/*/phase4/results; do
  [ -d "$r" ] && cp -rn "$r"/* results/kaggle/
done
find results/kaggle -name predictions.jsonl | sort | while read -r p; do echo "  $(wc -l < "$p") $p"; done

echo "== 2. phase4_analysis.sh"
bash runs/phase4_analysis.sh

echo "== 3. paired tests"
pair() {  # pair <a.jsonl> <b.jsonl> <name>
  if [ -f "$1" ] && [ -f "$2" ]; then
    $PY analysis/paired_compare.py --a "$1" --b "$2" --name "$3" --out_dir "$OUT" 2>&1 | tail -1
  else
    echo "  skip $3 (missing $( [ -f "$1" ] || echo "$1" ) $( [ -f "$2" ] || echo "$2" ))"
  fi
}
K=results/kaggle/finetuned
for s in splits_fa-eval fleurs-test cv-test kathbath-valid splits_fa-dev; do
  pair $K/$s/m5_merged_t4/predictions.jsonl $K/$s/m6_harmonized_t4/predictions.jsonl ${s}__m6_vs_m5
  pair results/modal/finetuned/$s/m1_full/predictions.jsonl $K/$s/m1_full_4ep_t4/predictions.jsonl ${s}__m1_full_4ep_vs_m1_full
done
# session D: every soup / beam row vs M1 full and vs M5 merged
for s in splits_fa-dev splits_fa-eval fleurs-test cv-test kathbath-valid; do
  for d in $K/$s/soup_*/ $K/$s/*_beam5_t4/; do
    [ -f "$d/predictions.jsonl" ] || continue
    r=$(basename "$d")
    pair results/modal/finetuned/$s/m1_full/predictions.jsonl "$d/predictions.jsonl" ${s}__${r}_vs_m1_full
    pair $K/$s/m5_merged_t4/predictions.jsonl "$d/predictions.jsonl" ${s}__${r}_vs_m5
  done
done
for s in splits_fa-eval fleurs-test kathbath-valid; do
  pair results/phase3/$s/mbr_lm_x/predictions.jsonl results/modal/finetuned/$s/m1_full/predictions.jsonl ${s}__m1_full_vs_phase3
  pair results/phase3/$s/mbr_lm_x/predictions.jsonl results/phase4_stack/$s/rover/predictions.jsonl ${s}__m1_full_rover_vs_phase3
done

echo "== 4. figures"
$PY figures/scripts/f4_finetune_by_set.py
$PY figures/scripts/f13_matrix_heatmap.py

echo "== 5. summary"
$PY - <<'EOF' | tee results/phase4_final_summary.txt
import json
from pathlib import Path
print("Phase 4 final summary (WER %, B - A in points [95% cluster CI])")
for f in sorted(Path("results/compare/phase4_final").glob("*.json")):
    if f.name.endswith(".run_info.json"):
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    lo, hi = (100 * x for x in d["diff_ci95"])
    print(f"{f.stem:45s} A {100 * d['wer_a']:6.2f}  B {100 * d['wer_b']:6.2f}  B-A {100 * d['diff_b_minus_a']:+6.2f} "
          f"[{lo:+.2f}, {hi:+.2f}]  ({d['n_utterances']} utt)")
print()
print("New rows vs zero-shot (results/compare/phase4):")
for f in sorted(Path("results/compare/phase4").glob("*_vs_tugstugi_a100.json")):
    if any(k in f.name for k in ("m6_", "4ep", "soup_", "beam5")):
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"{f.stem:55s} WER {100 * d['wer_b']:6.2f}  diff {100 * d['diff_b_minus_a']:+6.2f}")
print()
print("Speed (real-time factor on the T4, lower = faster; greedy M5 ~0.035-0.038):")
for m in sorted(Path("results/kaggle/finetuned").glob("*/*/metrics.json")):
    if any(k in m.parent.name for k in ("soup_", "beam5", "m5_merged")):
        x = json.loads(m.read_text(encoding="utf-8"))
        if x.get("last_run_rtf") is not None:
            print(f"{m.parent.parent.name + '/' + m.parent.name:70s} RTF {x['last_run_rtf']:.4f}")
EOF
echo "done: send results/phase4_final_summary.txt (or just say 'finish ran')"
