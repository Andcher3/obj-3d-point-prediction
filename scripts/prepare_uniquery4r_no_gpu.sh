#!/usr/bin/env bash
# Downloads/installations and CPU-only import/weight checks. Never invokes inference.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
if [[ ! -f /etc/network_turbo ]]; then
    echo "AutoDL academic acceleration script is missing" >&2
    exit 3
fi
source /etc/network_turbo >/dev/null 2>&1
echo "AUTODL_ACADEMIC_ACCELERATION_ENABLED"
if ! command -v aria2c >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends aria2
fi
export HF_HOME=$BASE/cache/huggingface
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=$BASE/cache/pip
export TMPDIR=$BASE/tmp
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$BASE"/{cache,tmp,weights,logs}
cd "$BASE"
echo "PREPARE_START $(date -u +%FT%TZ)"
df -h "$BASE"
/root/miniconda3/bin/python -u "$BASE/fetch_assets.py" --base "$BASE"
REPO=$(/root/miniconda3/bin/python -c "import json; print(json.load(open('$BASE/logs/asset-status.json'))['source_directory'])")
if [[ ! -x "$BASE/venv/bin/python" ]]; then
    /root/miniconda3/bin/python -m venv "$BASE/venv"
fi
PY=$BASE/venv/bin/python
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir numpy==1.26.4 opencv-python==4.11.0.86 'pillow>=10' 'einops>=0.7' 'trimesh>=4' 'matplotlib>=3.7' 'scipy>=1.10' huggingface_hub==0.34.4 viser
"$PY" -m pip install --no-cache-dir --no-deps -e "$REPO"
"$PY" -u "$BASE/download_and_check.py" --base "$BASE"
"$PY" -m pip freeze > "$BASE/logs/requirements-frozen.txt"
echo "PREPARE_COMPLETE_NO_INFERENCE $(date -u +%FT%TZ)"
df -h "$BASE"
