#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OBJECT_DIR ORIGINAL_DIR GPU_INDEX" >&2
  exit 2
fi
object_dir="$1"
original_dir="$2"
gpu="$3"
root=/root/autodl-tmp/projects
mf="$root/motionforesight"
base="$mf/outputs/pointodyssey_interface_pilot"
out="$base/$object_dir"
original="$base/$original_dir"
python=/root/autodl-tmp/conda_envs/motionforesight-e3/bin/python

export CUDA_VISIBLE_DEVICES="$gpu"
export MOTIONFORESIGHT_TOKENIZER_DIR=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl
export TRACKCRAFT_TOKENIZER_DIR="$MOTIONFORESIGHT_TOKENIZER_DIR"
export TRACKCRAFT_ATTN_Q_CHUNK=512
export MODELSCOPE_CACHE="$mf/assets/models_pretrained/wan_models"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$mf"
"$python" pointodyssey_interface_preflight_remote.py \
  --repo "$root/motionforesight_src" \
  --checkpoint "$mf/checkpoints/trainedon_ss100k/best.pt" \
  --base-checkpoint /root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors \
  --user-npz "$original/observed7_user.npz" \
  --trackcraft-dense "$original/observed7_trackcraft_dense.npz" \
  --query-json "$out/query_manifest.json" \
  --pseudo-gt-bundle "$out/pointodyssey_gt_bundle.npz" \
  --cache-npz "$out/p0_h_cache_aligned.npz" \
  --output-json "$out/formal_interface_aligned.json" \
  --ground-truth-source pointodyssey_v1.2_exact_rendered_tracks \
  --query-count 48 --fps 15 > "$out/motionforesight.log" 2>&1
echo "OBJECT_PREFLIGHT_COMPLETE:$object_dir"
