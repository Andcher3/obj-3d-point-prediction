#!/usr/bin/env bash
set -u
base=/root/autodl-tmp/uniquery4r-movif-experiment1
deployment=/root/autodl-tmp/uniquery4r-visual-review
python_bin="$deployment/venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$base" || exit 2
date -u +%FT%TZ > logs/gpu-run-started.txt
timeout 3600 "$python_bin" run_experiment1.py --base "$base" --deployment "$deployment" --gpu-enabled --query-approved --contexts 22
inference_status=$?
printf '%s\n' "$inference_status" > logs/gpu-inference-exit-code.txt
if [ "$inference_status" -ne 0 ]; then exit "$inference_status"; fi
date -u +%FT%TZ > logs/gpu-inference-completed.txt
"$python_bin" evaluate_gt.py --base "$base" --context 22
evaluation_status=$?
printf '%s\n' "$evaluation_status" > logs/gt-evaluation-exit-code.txt
exit "$evaluation_status"
