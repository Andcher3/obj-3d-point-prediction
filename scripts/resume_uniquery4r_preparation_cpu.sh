#!/usr/bin/env bash
# Restart failed transport only; reuse the active dependency installer.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
source /etc/network_turbo >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export TMPDIR=$BASE/tmp
export NO_PROXY=pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org,hf-mirror.com
export no_proxy="$NO_PROXY"
cd "$BASE"
/root/miniconda3/bin/python -u fetch_assets.py --base "$BASE" --checkpoint-endpoint mirror --connections 16
for check in $(seq 1 720); do
    if grep -q CPU_DEPENDENCIES_INSTALLED_NO_INFERENCE logs/environment-mirror.log; then
        PY=$BASE/venv/bin/python
        REPO=$BASE/source/UniQuery4R-95119742e61c64c27b09349035019c8cd7b41e62
        "$PY" -m pip install --no-cache-dir --timeout 60 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps -e "$REPO"
        "$PY" -u run_visual_review.py --base "$BASE" --check-only
        "$PY" -u download_and_check.py --base "$BASE"
        "$PY" -m pip check
        "$PY" -m pip freeze > logs/requirements-frozen.txt
        echo PREPARE_COMPLETE_NO_INFERENCE
        exit 0
    fi
    sleep 15
done
echo DEPENDENCY_INSTALL_WAIT_TIMEOUT_NO_INFERENCE >&2
exit 4
