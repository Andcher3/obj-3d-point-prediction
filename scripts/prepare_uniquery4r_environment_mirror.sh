#!/usr/bin/env bash
# Domestic package mirror, preserving PyTorch 2.8 / CUDA 12.8 dependencies.
set -euo pipefail
BASE=/root/autodl-tmp/uniquery4r-visual-review
source /etc/network_turbo >/dev/null 2>&1
# Domestic mirror should not be sent through the international proxy.
export NO_PROXY=pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org
export no_proxy="$NO_PROXY"
export TMPDIR=$BASE/tmp
export CUDA_VISIBLE_DEVICES=""
PY=$BASE/venv/bin/python
"$PY" -m pip install --no-cache-dir --timeout 60 torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -m pip install --no-cache-dir --timeout 60 numpy==1.26.4 opencv-python==4.11.0.86 'pillow>=10' 'einops>=0.7' 'trimesh>=4' 'matplotlib>=3.7' 'scipy>=1.10' huggingface_hub==0.34.4 viser --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -c 'import torch; assert torch.version.cuda == "12.8"; print("TORCH_CUDA_12_8_BUILD_VERIFIED", torch.__version__)'
echo CPU_DEPENDENCIES_INSTALLED_NO_INFERENCE
