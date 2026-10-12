#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/projects/TrackCraft3r
data=/root/autodl-tmp/ego4d-formal-e2
python=/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python
checkpoint=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors

export TRACKCRAFT_TOKENIZER_DIR=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl
export HF_ENDPOINT=https://hf-mirror.com
export TRACKCRAFT_ATTN_Q_CHUNK=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$data/trackcraft_outputs" "$data/formal_bundles"
cd "$repo"

for video_id in "$@"; do
  dense="$data/trackcraft_outputs/${video_id}-dense-chunk1024.npz"
  log="$data/trackcraft_outputs/${video_id}-dense-chunk1024.log"
  input="$data/trackcraft_inputs/$video_id/input.npz"
  query="$data/query_manifests/${video_id}_query_manifest.json"
  bundle="$data/formal_bundles/$video_id.npz"

  if [[ ! -f "$dense" ]]; then
    "$python" run_trackcraft_release_inference.py \
      --checkpoint_path "$checkpoint" \
      --input_npz "$input" \
      --output_npz "$dense" \
      --num_frames 12 --frame_stride 1 >"$log" 2>&1
  fi
  "$python" build_trackcraft_formal_e2_bundle.py \
    --dense-npz "$dense" \
    --camera-npz "$input" \
    --query-json "$query" \
    --output-npz "$bundle" | tee "$data/formal_bundles/${video_id}.json"
done
