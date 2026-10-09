#!/usr/bin/env bash
# Phase 4, first night: M1 = tugstugi fine-tuned on DL Sprint train (FA split,
# cleaned text), two ways, each scored on every test set with the frozen harness.
#   1) clean the training text (strip [মিউজিক]-style tags, drop 67 chunks with
#      non-Bangla script)                                        (seconds)
#   2) M1 LoRA: train 2 epochs (~4.5 h), then dev / eval / FLEURS / Kathbath / CV (~1.5 h)
#   3) M1 decoder-only: train 2 epochs (~2.5 h), then the same test sets (~1.3 h)
#   4) error types + paired comparisons vs tugstugi zero-shot, summary (minutes)
# Run from the repo root, in Git Bash, with games/overlays/other GPU apps closed:
#     bash runs/phase4_finetune.sh
#
# Breakable: stop any time (Ctrl+C, shutdown, power cut) and run the same command
# again. Finished steps are skipped (marker files in logs/phase4/done/); training
# continues from its last checkpoint (saved every 100 steps and on Ctrl+C: press it
# once and wait for "Stopped", ~15 s; a second press quits without saving).
# A failed step is logged and skipped; it is retried on the next run.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=./Scripts/python.exe
LOG=logs/phase4
mkdir -p "$LOG/done"
STATUS="$LOG/status.txt"
: > "$STATUS"
trap 'echo; echo "Stopped. Progress is saved; run  bash runs/phase4_finetune.sh  again to continue."; exit 130' INT

step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "$LOG/done/$name" ]; then
    echo "SKIP    $name (already done)" | tee -a "$STATUS"
    return
  fi
  echo "=== $(date '+%a %H:%M') START $name   (full log: $LOG/$name.log)"
  # tee ignores Ctrl+C, so the training script can finish its step and save before the log pipe closes
  "$@" 2>&1 | (trap '' INT; tee -a "$LOG/$name.log")
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
FT=results/finetuned
# set name (as in results/zeroshot) -> manifest
declare -A SETS=(
  [splits_fa-dev]=data/splits_fa/dev.csv
  [splits_fa-eval]=data/splits_fa/eval.csv
  [fleurs-test]=data/external/fleurs/test.csv
  [kathbath-valid]=data/external/kathbath/valid.csv
  [cv-test]=data/external/cv/test.csv
)
ORDER="splits_fa-dev splits_fa-eval fleurs-test kathbath-valid cv-test"

# score <run name> <eval_zero_shot model args...>: every test set + error types + paired vs tugstugi
score() {
  local run=$1; shift
  for s in $ORDER; do
    step "test_${run}_$s"  $PY pipeline/eval_zero_shot.py --manifest "${SETS[$s]}" "$@" --name "$run" --out_dir $FT
    step "types_${run}_$s" $PY analysis/error_types.py --pred "$FT/$s/$run/predictions.jsonl"
    step "cmp_${run}_$s"   $PY analysis/paired_compare.py --a "$ZS/$s/$TUGP/predictions.jsonl" \
                               --b "$FT/$s/$run/predictions.jsonl" --name "${s}_${run}_vs_tugstugi"
  done
}

# 1) Training text cleanup; dev and eval references are untouched
step clean_train   $PY pipeline/data/clean_train_text.py

# 2) M1 LoRA (r 32, all attention + MLP projections, lr 2e-4, batch 8x4, 2 epochs = ~956 steps)
step train_m1_lora $PY pipeline/11_finetune.py --method lora --train_manifest data/splits_fa/train_clean.csv \
                       --out_dir checkpoints/m1_lora
[ -f "$LOG/done/train_m1_lora" ] && score m1_lora --model $TUG --adapter checkpoints/m1_lora/final

# 3) M1 decoder-only (encoder frozen, lr 1e-5, same batches and epochs)
step train_m1_decoder $PY pipeline/11_finetune.py --method decoder --train_manifest data/splits_fa/train_clean.csv \
                          --out_dir checkpoints/m1_decoder
[ -f "$LOG/done/train_m1_decoder" ] && score m1_decoder --model checkpoints/m1_decoder/final

# 4) Summary (always rebuilt)
rm -f "$LOG/done/summary"
step summary       $PY analysis/collect_results.py

echo
echo "================ DONE $(date) ================"
cat "$STATUS"
echo "Next: share the status list above and results/summary.md"
