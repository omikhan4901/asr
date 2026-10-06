#!/usr/bin/env bash
# Phase 1 batch: split -> test-set downloads -> 12-best + plateau analysis -> zero-shot baselines.
# Run AFTER pipeline/01_segment.py has finished, from the repo root, in Git Bash:
#     bash runs/day1.sh
#
# Breakable: stop any time (Ctrl+C, shutdown, power cut) and run the same command
# again. Finished steps are skipped (marker files in logs/day1/done/), and the
# long steps continue from the last finished batch inside the step.
# A failed step is logged and skipped; it is retried on the next run.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=./Scripts/python.exe
LOG=logs/day1
mkdir -p "$LOG/done"
STATUS="$LOG/status.txt"
: > "$STATUS"
trap 'echo; echo "Stopped. Progress is saved; run  bash runs/day1.sh  again to continue."; exit 130' INT

step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "$LOG/done/$name" ]; then
    echo "SKIP    $name (already done)" | tee -a "$STATUS"
    return
  fi
  echo "=== $(date '+%a %H:%M') START $name   (full log: $LOG/$name.log)"
  # Show live output (progress bars) on screen and keep a copy in the log.
  "$@" 2>&1 | tee -a "$LOG/$name.log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    touch "$LOG/done/$name"
    echo "OK      $name" | tee -a "$STATUS"
  else
    echo "FAILED  $name   (see $LOG/$name.log)" | tee -a "$STATUS"
  fi
}

TUG=bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium

# 1) Train/dev/eval split of the forced-aligned chunks (seconds)
step split            $PY pipeline/02_split.py

# 2) Public test sets (downloads; FLEURS ~0.8 GB, Kathbath valid ~0.5 GB)
step prep_fleurs      $PY pipeline/data/prepare_hf.py --preset fleurs --split test
step prep_kathbath    $PY pipeline/data/prepare_hf.py --preset kathbath --split valid

# 3) Baseline 1-best on our long-form eval (sanity check, ~40 min)
step zs_eval_tugstugi $PY pipeline/eval_zero_shot.py --manifest data/splits_fa/eval.csv --model $TUG

# 4) 12-best on the long-form eval (the published 499A decoding setup; several hours)
step nbest_eval       $PY pipeline/03_generate_nbest.py --split_csv data/splits_fa/eval.csv \
                          --output_jsonl data/nbest_fa/eval_nbest.jsonl --num_beams 12 --temperature 0.7 --batch_size 2

# 5) Plateau anatomy on that 12-best (minutes, CPU)
step agreement_eval   $PY analysis/agreement.py --nbest data/nbest_fa/eval_nbest.jsonl \
                          --out_dir results/agreement/eval_tugstugi

# 6) Zero-shot baselines: every model x every test set
MODELS="$TUG openai/whisper-large-v3-turbo arijitx/wav2vec2-xls-r-300m-bengali ai4bharat/indicwav2vec_v1_bengali bangla-speech-processing/BanglaASR openai/whisper-large-v3"
for m in $MODELS; do
  slug=${m//\//__}
  step "zs_fleurs__$slug"   $PY pipeline/eval_zero_shot.py --manifest data/external/fleurs/test.csv --model $m
  step "zs_kathbath__$slug" $PY pipeline/eval_zero_shot.py --manifest data/external/kathbath/valid.csv --model $m
  # Generic multilingual Whisper (large-v3, turbo) writes Bangla in mixed scripts
  # zero-shot (~65-95% WER); FLEURS/Kathbath rows make that point, so they skip
  # the 17 h long-form eval (saves ~3-4 GPU-hours). Tugstugi ran it in step 3.
  case "$m" in
    "$TUG"|openai/*) ;;
    *) step "zs_eval__$slug" $PY pipeline/eval_zero_shot.py --manifest data/splits_fa/eval.csv --model $m ;;
  esac
done

# 7) One table of everything
rm -f "$LOG/done/summary"   # always rebuild the table
step summary          $PY analysis/collect_results.py

echo
echo "================ DONE $(date) ================"
cat "$STATUS"
echo "Next: share the status list and results/summary.md"
