#!/usr/bin/env bash
# CPU-only validation after the independently running mirror install completes.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for check in $(seq 1 720); do
    if grep -q CPU_DEPENDENCIES_INSTALLED_NO_INFERENCE "$BASE/logs/environment-mirror.log"; then
        export PYTHONPATH=$BASE/source/UniQuery4R-95119742e61c64c27b09349035019c8cd7b41e62
        "$BASE/venv/bin/python" -u "$BASE/check_environment_without_weights.py" --base "$BASE"
        "$BASE/venv/bin/python" -u "$BASE/run_visual_review.py" --base "$BASE" --check-only
        "$BASE/venv/bin/python" -u "$BASE/test_render_without_model.py" --base "$BASE" --renderer "$BASE/run_visual_review.py"
        "$BASE/venv/bin/python" -m pip check
        "$BASE/venv/bin/python" -m pip freeze > "$BASE/logs/requirements-frozen.txt"
        echo CPU_INTERFACE_COMPLETE_NO_INFERENCE
        exit 0
    fi
    sleep 15
done
echo CPU_INTERFACE_WAIT_TIMEOUT_NO_INFERENCE >&2
exit 4
