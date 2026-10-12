#!/usr/bin/env bash
# Keep the existing environment; resume package data with official hash checks.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
source /etc/network_turbo >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export TMPDIR=$BASE/tmp
export NO_PROXY=pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org,hf-mirror.com
export no_proxy="$NO_PROXY"
/root/miniconda3/bin/python -u "$BASE/prepare_cuda_wheels.py" --base "$BASE"
PY=$BASE/venv/bin/python
# Explicit local paths prevent pip preferring a same-version remote wheel.
"$PY" -m pip install --no-cache-dir --no-index --no-deps "$BASE"/wheelhouse/*.whl
"$PY" -m pip install --no-cache-dir --timeout 60 --find-links "$BASE/wheelhouse" torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -m pip install --no-cache-dir --timeout 60 numpy==1.26.4 opencv-python==4.11.0.86 'pillow>=10' 'einops>=0.7' 'trimesh>=4' 'matplotlib>=3.7' 'scipy>=1.10' huggingface_hub==0.34.4 viser --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -c 'import torch; assert torch.version.cuda == "12.8"; print("TORCH_CUDA_12_8_BUILD_VERIFIED", torch.__version__)'
echo CPU_DEPENDENCIES_INSTALLED_NO_INFERENCE
