#!/usr/bin/env bash
# Only run after uploaded weights pass the official streaming hash.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=$BASE/tmp
export NO_PROXY=pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org,hf-mirror.com
export no_proxy="$NO_PROXY"
source /etc/network_turbo >/dev/null 2>&1
cd "$BASE"
for check in $(seq 1 720); do
    if grep -q CPU_INTERFACE_COMPLETE_NO_INFERENCE logs/interface-cpu-parallel.log; then
        PY=$BASE/venv/bin/python
        REPO=$BASE/source/UniQuery4R-95119742e61c64c27b09349035019c8cd7b41e62
        "$PY" -m pip install --no-cache-dir --timeout 60 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps -e "$REPO"
        "$PY" -u download_and_check.py --base "$BASE"
        "$PY" -m pip check
        "$PY" -m pip freeze > logs/requirements-frozen.txt
        echo NO_CARD_PREPARATION_COMPLETE_AWAITING_GPU_ENABLE
        exit 0
    fi
    if grep -q 'Traceback (most recent call last)\|CPU_INTERFACE_WAIT_TIMEOUT' logs/interface-cpu-parallel.log; then
        echo CPU_INTERFACE_CHECK_FAILED_NO_INFERENCE >&2
        exit 3
    fi
    sleep 15
done
echo CPU_INTERFACE_WAIT_TIMEOUT_NO_INFERENCE >&2
exit 4
