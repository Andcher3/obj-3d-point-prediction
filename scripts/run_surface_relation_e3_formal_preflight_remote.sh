#!/usr/bin/env bash
set -euo pipefail

motion_repo=/root/autodl-tmp/projects/motionforesight
track_repo=/root/autodl-tmp/projects/TrackCraft3r
data_root=/root/autodl-tmp/ego4d-formal-e2
output_root="$motion_repo/outputs/e3_formal"
python=/root/autodl-tmp/conda_envs/motionforesight-e3/bin/python
base_checkpoint=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
checkpoint="$motion_repo/checkpoints/trainedon_ss100k/best.pt"

export MOTIONFORESIGHT_TOKENIZER_DIR=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl
export TRACKCRAFT_TOKENIZER_DIR="$MOTIONFORESIGHT_TOKENIZER_DIR"
export TRACKCRAFT_ATTN_Q_CHUNK=512
export MODELSCOPE_CACHE="$motion_repo/assets/models_pretrained/wan_models"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

items=(
  "cb78dba8-6647-4ec7-875c-6ba1548f8312|67|1492.0"
  "356ed369-7023-4660-904b-be73823a21d3|67|1492.0"
  "65165142-b15e-4bba-a9de-d5174ec46597|102|1492.0"
  "9069bd26-0139-4e88-9957-9f9489b35a91|87|0.0210286"
  "4d053322-1eef-4a69-8201-932c7b82fb03|95|892.0"
)

for item in "${items[@]}"; do
  IFS='|' read -r video_id t0_index clip_start <<<"$item"
  if [[ $# -gt 0 ]]; then
    selected=0
    for requested in "$@"; do
      [[ "$video_id" == "$requested" ]] && selected=1
    done
    [[ $selected -eq 0 ]] && continue
  fi

  out="$output_root/$video_id"
  mkdir -p "$out"
  indices=()
  for ((index=t0_index-6; index<=t0_index; index++)); do
    indices+=("$index")
  done

  if [[ ! -f "$out/observed7_user.npz" ]]; then
    "$python" "$motion_repo/surface_relation_e3_prepare_formal_input.py" \
      --frames-dir "$data_root/frames_15fps/$video_id" \
      --frame-indices "${indices[@]}" \
      --clip-start-seconds "$clip_start" --fps 15 \
      --output-npz "$out/observed7_user.npz" \
      --output-json "$out/observed7_input.json"
  fi

  if [[ ! -f "$out/observed7_trackcraft_dense.npz" ]]; then
    (
      cd "$track_repo"
      "$python" run_trackcraft_release_inference.py \
        --checkpoint_path "$base_checkpoint" \
        --input_npz "$out/observed7_user.npz" \
        --output_npz "$out/observed7_trackcraft_dense.npz" \
        --num_frames 7 --frame_stride 1
    )
  fi

  (
    cd "$motion_repo"
    "$python" surface_relation_e3_formal_preflight.py \
      --repo "$motion_repo" \
      --checkpoint "$checkpoint" \
      --base-checkpoint "$base_checkpoint" \
      --user-npz "$out/observed7_user.npz" \
      --trackcraft-dense "$out/observed7_trackcraft_dense.npz" \
      --query-json "$data_root/query_manifests/${video_id}_query_manifest.json" \
      --pseudo-gt-bundle "$data_root/formal_bundles/${video_id}.npz" \
      --cache-npz "$out/p0_h_cache_aligned.npz" \
      --output-json "$out/formal_interface_aligned.json" \
      --query-count 48 --fps 15
  )
done
