#!/usr/bin/env bash
# Phase 1 batch: split -> test-set downloads -> 12-best + plateau analysis -> zero-shot baselines.
# Run AFTER pipeline/01_segment.py has finished, from the repo root, in Git Bash:
#     bash runs/day1.sh
# Every step is resumable: if anything stops, run the same command again.
# A failed step is logged and skipped; the rest continue. Summary at the end.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8
PY=./Scripts/python.exe
LOG=logs/day1
mkdir -p "$LOG"
STATUS="$LOG/status.txt"
: > "$STATUS"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "=== $(date '+%a %H:%M') START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
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
  [ "$m" != "$TUG" ] && step "zs_eval__$slug" $PY pipeline/eval_zero_shot.py --manifest data/splits_fa/eval.csv --model $m
done

# 7) One table of everything
step summary          $PY analysis/collect_results.py

echo
echo "================ DONE $(date) ================"
cat "$STATUS"
echo "Next: share the status list and results/summary.md"
