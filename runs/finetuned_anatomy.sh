#!/usr/bin/env bash
# E11: the N-best anatomy (E3 + E5 + E23) after fine-tuning, on M1 full's sampled 12-best
# (Kaggle session B, exact candidate scores) and its A100 greedy output. Same scripts and
# settings as the zero-shot anatomy in runs/rescoring_audit.sh. CPU only.
#     bash runs/finetuned_anatomy.sh
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || { [ -x ./Scripts/python.exe ] && echo ./Scripts/python.exe || echo python; } )}  # Linux/Mac venv, Windows venv, or PATH
NB=data/kaggle_out/session_B/outputs/nbest
FT=results/modal/finetuned
LOG=logs/finetuning/e11
mkdir -p "$LOG"

anatomy() {  # anatomy <set> <name>
  local nb="$NB/$1__m1_full_scored.jsonl" pred="$FT/$1/m1_full/predictions.jsonl" n=$2
  $PY analysis/agreement.py --nbest "$nb" --out_dir "results/agreement/$n" > "$LOG/agreement_$n.log" 2>&1 || echo "FAILED agreement_$n"
  $PY analysis/error_types.py --pred "$pred" --nbest "$nb" --out_dir results/error_types/finetuned_anatomy > "$LOG/types_$n.log" 2>&1 || echo "FAILED types_$n"
  $PY analysis/agreement_signal.py --pred "$pred" --nbest "$nb" --name "$n" > "$LOG/signal_$n.log" 2>&1 || echo "FAILED signal_$n"
  for m in mbr oracle rank0; do
    $PY analysis/paired_compare.py --a "$pred" --b "$nb:$m" --name "${n}_${m}_vs_greedy" > "$LOG/cmp_${m}_$n.log" 2>&1 || echo "FAILED cmp_${m}_$n"
  done
  echo "done $n"
}

anatomy splits_fa-dev  dev_sampled_m1_full
anatomy splits_fa-eval eval_sampled_m1_full
anatomy fleurs-test    fleurs_m1_full
anatomy kathbath-valid kathbath_m1_full
