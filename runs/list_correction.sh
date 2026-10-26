#!/usr/bin/env bash
# List correction: raising the selection ceiling without retraining (PLAN E7, E8, E23, E24).
#   1. tugstugi greedy on the cleaned TRAIN split (GPU, ~1.5-2 h): the errors the
#      convention rewriter learns from
#   2. candidate pools = sampled 12-best + the greedy hypothesis with its exact
#      score, for dev, eval, FLEURS, Kathbath (GPU, ~15 min)
#   3. n-gram LM rescoring, LM-weighted MBR, word-level combination, convention
#      rewriter, N-best expansion; all tuned on dev, applied to every set (CPU, ~15 min)
#   4. error types of every list correction output (what got fixed), then a summary
# Run from the repo root, in Git Bash:
#     bash runs/list_correction.sh
#
# Breakable: stop any time (Ctrl+C, shutdown, power cut) and run the same command
# again. Finished steps are skipped (marker files in logs/list_correction/done/); the GPU
# steps continue from the last finished utterance.
# A failed step is logged and skipped; it is retried on the next run.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || { [ -x ./Scripts/python.exe ] && echo ./Scripts/python.exe || echo python; } )}  # Linux/Mac venv, Windows venv, or PATH
LOG=logs/list_correction
mkdir -p "$LOG/done" data/list_correction
STATUS="$LOG/status.txt"
: > "$STATUS"
trap 'echo; echo "Stopped. Progress is saved; run  bash runs/list_correction.sh  again to continue."; exit 130' INT

step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "$LOG/done/$name" ]; then
    echo "SKIP    $name (already done)" | tee -a "$STATUS"
    return
  fi
  echo "=== $(date '+%a %H:%M') START $name   (full log: $LOG/$name.log)"
  "$@" 2>&1 | tee -a "$LOG/$name.log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    touch "$LOG/done/$name"
    echo "OK      $name" | tee -a "$STATUS"
  else
    echo "FAILED  $name   (see $LOG/$name.log)" | tee -a "$STATUS"
  fi
}

TUG=bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium
TUGP=bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium
ZS=results/zeroshot
P3=data/list_correction

# 1. greedy on train (same harness and batch size as every zero-shot row)
step zs_train $PY pipeline/eval_zero_shot.py --manifest data/splits_fa/train_clean.csv --model $TUG --set_name splits_fa-train

# 2. pools: <name> <N-best> <manifest> <greedy set>
pool() {
  step "pool_$1" $PY pipeline/rescore_nbest.py --input_jsonl "$2" --manifest "$3" \
      --output_jsonl "$P3/$1_pool.jsonl" --add_greedy "$ZS/$4/$TUGP/predictions.jsonl"
}
pool dev      data/nbest_fa/dev_sampled.jsonl       data/splits_fa/dev.csv          splits_fa-dev
pool eval     data/nbest_fa/eval_sampled.jsonl      data/splits_fa/eval.csv         splits_fa-eval
pool fleurs   data/nbest_ext/fleurs_tugstugi.jsonl  data/external/fleurs/test.csv   fleurs-test
pool kathbath data/nbest_ext/kathbath_tugstugi.jsonl data/external/kathbath/valid.csv kathbath-valid

# 3. all list correction methods (needs every step above)
ready=1
for s in zs_train pool_dev pool_eval pool_fleurs pool_kathbath; do
  [ -f "$LOG/done/$s" ] || ready=0
done
if [ $ready -eq 1 ]; then
  step ceiling $PY pipeline/12_raise_ceiling.py
else
  echo "WAIT    ceiling (needs zs_train and all pools; run again)" | tee -a "$STATUS"
fi

# 4. error types of every output, per set
if [ -f "$LOG/done/ceiling" ]; then
  for s in splits_fa-dev splits_fa-eval fleurs-test kathbath-valid; do
    step "types_$s" $PY analysis/error_types.py --out_dir results/error_types/list_correction \
        --pred results/list_correction/$s/*/predictions.jsonl
  done
  echo
  echo "=== list correction summary (WER %, paired difference vs greedy with 95% CI) ==="
  $PY - <<'EOF'
import pandas as pd
d = pd.read_csv("results/list_correction/summary.csv")
methods = ["am", "lm", "mbr_lm", "rover", "rewrite", "rewrite_sel", "lm_x", "mbr_lm_x"]
for _, r in d.iterrows():
    print(f"\n{r['set']}  ({r['n_utterances']} utt, {r['n_groups']} groups)  greedy {100 * r['greedy_wer']:.2f}")
    for m in methods:
        print(f"  {m:12s} {100 * r[m + '_wer']:6.2f}   {100 * r[m + '_diff']:+.2f} "
              f"[{100 * r[m + '_ci_lo']:+.2f}, {100 * r[m + '_ci_hi']:+.2f}]  P(better) {r[m + '_p_better']:.3f}")
    print(f"  oracle {100 * r['oracle_pool_wer']:.2f} -> {100 * r['oracle_x_wer']:.2f} after expansion; "
          f"coverage miss {100 * r['coverage_miss_pool']:.2f} -> {100 * r['coverage_miss_x']:.2f}%; "
          f"selection-proof share of greedy errors {100 * r['greedy_errors_selection_proof_pool']:.1f} -> "
          f"{100 * r['greedy_errors_selection_proof_x']:.1f}%")
EOF
fi

echo
echo "=== Status ==="
cat "$STATUS"
echo "Keep the summary above (or the last 40 lines of this window) for the analysis."
