#!/usr/bin/env bash
# Independent CPU-side dependency installation while official assets download.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
source /etc/network_turbo >/dev/null 2>&1
export TMPDIR=$BASE/tmp
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$BASE/tmp" "$BASE/logs"
/root/miniconda3/bin/python -m venv "$BASE/venv"
PY=$BASE/venv/bin/python
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir numpy==1.26.4 opencv-python==4.11.0.86 'pillow>=10' 'einops>=0.7' 'trimesh>=4' 'matplotlib>=3.7' 'scipy>=1.10' huggingface_hub==0.34.4 viser
echo CPU_DEPENDENCIES_INSTALLED_NO_INFERENCE
