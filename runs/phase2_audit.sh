#!/usr/bin/env bash
# Phase 2: reference audit and error types (GPU part + the analyses that read it).
#   - exact candidate scores for the Phase 1 eval 12-best, then MBR recomputed
#   - dev set: greedy and sampled 12-best (decoding choices are made on dev)
#   - agreement anatomy on more test sets and a second model (E17): tugstugi on
#     FLEURS and Kathbath, BanglaASR (Whisper-small) on eval and FLEURS
#   - Common Voice test set: zero-shot rows
#   (Deterministic beam N-best is left out: in transformers 5.15, Whisper
#    generate() with num_return_sequences=12 does not return the 12 best beams.)
#   - after each N-best: anatomy (E3), error types by agreement (E5),
#     agreement signal (E23), paired comparisons; at the end E21 + E20 + summary.
# Run from the repo root, in Git Bash:
#     bash runs/phase2_audit.sh
#
# Breakable: stop any time (Ctrl+C, shutdown, power cut) and run the same command
# again. Finished steps are skipped (marker files in logs/phase2/done/), and the
# long steps continue from the last finished batch inside the step.
# A failed step is logged and skipped; it is retried on the next run.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=./Scripts/python.exe
LOG=logs/phase2
mkdir -p "$LOG/done"
STATUS="$LOG/status.txt"
: > "$STATUS"
trap 'echo; echo "Stopped. Progress is saved; run  bash runs/phase2_audit.sh  again to continue."; exit 130' INT

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
SMALL=bangla-speech-processing/BanglaASR
ZS=results/zeroshot
TUGP=bengaliAI__tugstugi_bengaliai-regional-asr_whisper-medium
SMALLP=bangla-speech-processing__BanglaASR
NB=data/nbest_fa
NX=data/nbest_ext
mkdir -p "$NX"

# anatomy <nbest> <greedy predictions> <name>: E3 + E5 + E23 for one N-best
anatomy() {
  step "agreement_$3"   $PY analysis/agreement.py --nbest "$1" --out_dir "results/agreement/$3"
  step "types_$3"       $PY analysis/error_types.py --pred "$2" --nbest "$1"
  step "signal_$3"      $PY analysis/agreement_signal.py --pred "$2" --nbest "$1" --name "$3"
  step "cmp_mbr_$3"     $PY analysis/paired_compare.py --a "$2" --b "$1:mbr"    --name "$3_mbr_vs_greedy"
  step "cmp_oracle_$3"  $PY analysis/paired_compare.py --a "$2" --b "$1:oracle" --name "$3_oracle_vs_greedy"
  step "cmp_rank0_$3"   $PY analysis/paired_compare.py --a "$2" --b "$1:rank0"  --name "$3_rank0_vs_greedy"
}

# 1) Common Voice test manifest (seconds; reads the extracted corpus, copies no audio)
step prep_cv          $PY pipeline/data/prepare_cv.py --split test

# 1b) Exact candidate scores for the Phase 1 eval 12-best (~50 min). Its stored
#     "logprob" came from the wrong beams (transformers no longer returns beam
#     scores for Whisper), so MBR there is wrong; texts, oracle and agreement are fine.
#     Writes a new file; the original eval_nbest.jsonl is kept.
step rescore_eval     $PY pipeline/rescore_nbest.py --input_jsonl $NB/eval_nbest.jsonl \
                          --manifest data/splits_fa/eval.csv --output_jsonl $NB/eval_sampled.jsonl
anatomy $NB/eval_sampled.jsonl $ZS/splits_fa-eval/$TUGP/predictions.jsonl eval_sampled_tugstugi

# 2) Dev set: greedy + sampled 12-best (~30 min). Decoding choices are made here, not on eval.
step zs_dev_tugstugi  $PY pipeline/eval_zero_shot.py --manifest data/splits_fa/dev.csv --model $TUG
step nbest_dev_sampled $PY pipeline/03_generate_nbest.py --split_csv data/splits_fa/dev.csv \
                          --output_jsonl $NB/dev_sampled.jsonl --decoding sampled --num_beams 12 --temperature 0.7 --batch_size 2
anatomy $NB/dev_sampled.jsonl $ZS/splits_fa-dev/$TUGP/predictions.jsonl dev_sampled_tugstugi

# 3) E17: tugstugi 12-best on FLEURS (~35 min) and Kathbath-valid (~40 min)
step nbest_fleurs_tugstugi   $PY pipeline/03_generate_nbest.py --split_csv data/external/fleurs/test.csv \
                          --output_jsonl $NX/fleurs_tugstugi.jsonl --decoding sampled --num_beams 12 --temperature 0.7 --batch_size 2
anatomy $NX/fleurs_tugstugi.jsonl $ZS/fleurs-test/$TUGP/predictions.jsonl fleurs_tugstugi
step nbest_kathbath_tugstugi $PY pipeline/03_generate_nbest.py --split_csv data/external/kathbath/valid.csv \
                          --output_jsonl $NX/kathbath_tugstugi.jsonl --decoding sampled --num_beams 12 --temperature 0.7 --batch_size 2
anatomy $NX/kathbath_tugstugi.jsonl $ZS/kathbath-valid/$TUGP/predictions.jsonl kathbath_tugstugi

# 4) E17 second model: BanglaASR (Whisper-small fine-tune) 12-best on eval (~1 h) and FLEURS (~15 min)
step nbest_eval_banglaasr   $PY pipeline/03_generate_nbest.py --split_csv data/splits_fa/eval.csv --model_id $SMALL \
                          --output_jsonl $NX/eval_banglaasr.jsonl --decoding sampled --num_beams 12 --temperature 0.7 --batch_size 2
anatomy $NX/eval_banglaasr.jsonl $ZS/splits_fa-eval/$SMALLP/predictions.jsonl eval_banglaasr
step nbest_fleurs_banglaasr $PY pipeline/03_generate_nbest.py --split_csv data/external/fleurs/test.csv --model_id $SMALL \
                          --output_jsonl $NX/fleurs_banglaasr.jsonl --decoding sampled --num_beams 12 --temperature 0.7 --batch_size 2
anatomy $NX/fleurs_banglaasr.jsonl $ZS/fleurs-test/$SMALLP/predictions.jsonl fleurs_banglaasr

# 5) Common Voice test, zero-shot (~1.5 h). Generic Whisper (large-v3, turbo) skipped:
#    it writes mixed scripts (FLEURS/Kathbath rows already show that).
for m in $TUG ai4bharat/indicwav2vec_v1_bengali $SMALL arijitx/wav2vec2-xls-r-300m-bengali; do
  step "zs_cv__${m//\//__}" $PY pipeline/eval_zero_shot.py --manifest data/external/cv/test.csv --model $m
done

# 6) Tables that cover everything (always rebuilt)
rm -f "$LOG/done/types_all" "$LOG/done/fingerprint" "$LOG/done/summary"
step types_all        $PY analysis/error_types.py --all_zeroshot
step fingerprint      $PY analysis/convention_fingerprint.py
step summary          $PY analysis/collect_results.py

echo
echo "================ DONE $(date) ================"
cat "$STATUS"
echo "Next: share the status list and results/summary.md"
