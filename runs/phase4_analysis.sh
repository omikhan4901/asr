#!/usr/bin/env bash
# Phase 4 analysis (CPU only, minutes): every scored Phase 4 model vs zero-shot tugstugi re-scored
# on the same GPU (tugstugi_a100), on every test set that is complete for both.
#   paired cluster bootstrap  -> results/compare/phase4/<set>__<run>_vs_tugstugi_a100.json
#   error types (variant / other) -> results/error_types/phase4/<set>/<run>/summary.json + summary.csv
#   GPU check (same model, two GPUs) -> results/compare/phase4/gpu_check.json
# Re-run after new rows arrive (e.g. LoRA's Kathbath/CV from Kaggle): everything is recomputed.
#   bash runs/phase4_analysis.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8
PY=./Scripts/python.exe
R=results/modal
BASE=tugstugi_a100
mkdir -p results/compare/phase4 results/error_types/phase4

n_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# Kaggle T4 rows (results/kaggle, names end in _t4) count too: T4 = A100 within 0.01 WER points
# on dev / FLEURS / Kathbath (results/compare/gpu_t4/).
preds=()
for set_dir in $R/finetuned/*/ results/kaggle/finetuned/*/; do
  set=$(basename "$set_dir")
  base=$R/zeroshot/$set/$BASE/predictions.jsonl
  [ -f "$base" ] || continue
  [[ " ${preds[*]} " == *" $base "* ]] || preds+=("$base")
  for run_dir in "$set_dir"*/; do
    run=$(basename "$run_dir")
    p=$run_dir/predictions.jsonl
    if [ "$(n_lines "$p")" != "$(n_lines "$base")" ]; then
      echo "skip $set/$run: $(n_lines "$p") of $(n_lines "$base") utterances"
      continue
    fi
    preds+=("$p")
    $PY analysis/paired_compare.py --a "$base" --b "$p" --name "${set}__${run}_vs_${BASE}" \
        --out_dir results/compare/phase4 > /dev/null || echo "FAILED paired $set/$run"
  done
done
$PY analysis/error_types.py --pred "${preds[@]}" --out_dir results/error_types/phase4 > results/error_types/phase4/log.txt 2>&1 \
    || echo "FAILED error types (see results/error_types/phase4/log.txt)"

# Same model on two GPUs: A100 re-score vs the 3060 rows of Phases 1-3.
$PY - <<'EOF'
import json
from pathlib import Path
out = {}
for a100 in sorted(Path("results/modal/zeroshot").glob("*/tugstugi_a100/predictions.jsonl")):
    s = a100.parts[-3]
    loc = Path("results/zeroshot") / s / "bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium/predictions.jsonl"
    if not loc.exists():
        continue
    A = {json.loads(l)["utt_id"]: json.loads(l)["hyp"] for l in a100.open(encoding="utf-8")}
    B = {json.loads(l)["utt_id"]: json.loads(l)["hyp"] for l in loc.open(encoding="utf-8")}
    common = A.keys() & B.keys()
    same = sum(A[u] == B[u] for u in common)
    out[s] = {"utterances": len(common), "identical_hypotheses": same, "identical_share": round(same / len(common), 4)}
Path("results/compare/phase4/gpu_check.json").write_text(json.dumps(out, indent=2))
print("GPU check (A100 vs 3060, tugstugi greedy):", json.dumps(out))
EOF
echo "done"
