#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects
mf="$root/motionforesight"
tc="$root/TrackCraft3r"
out="$mf/outputs/pointodyssey_interface_pilot/export_r4"
python=/root/autodl-tmp/conda_envs/motionforesight-e3/bin/python
base_checkpoint=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
checkpoint="$mf/checkpoints/trainedon_ss100k/best.pt"

export MOTIONFORESIGHT_TOKENIZER_DIR=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl
export TRACKCRAFT_TOKENIZER_DIR="$MOTIONFORESIGHT_TOKENIZER_DIR"
export TRACKCRAFT_ATTN_Q_CHUNK=512
export MODELSCOPE_CACHE="$mf/assets/models_pretrained/wan_models"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$out/observed7_user.npz" ]]; then
  cd "$mf"
  "$python" surface_relation_e3_prepare_formal_input.py \
    --frames-dir "$out/frames" --frame-indices 0 1 2 3 4 5 6 \
    --clip-start-seconds 0 --fps 15 \
    --output-npz "$out/observed7_user.npz" \
    --output-json "$out/observed7_input.json" > "$out/da3.log" 2>&1
fi
echo "DA3_COMPLETE"

if [[ ! -f "$out/observed7_trackcraft_dense.npz" ]]; then
  cd "$tc"
  "$python" run_trackcraft_release_inference.py \
    --checkpoint_path "$base_checkpoint" \
    --input_npz "$out/observed7_user.npz" \
    --output_npz "$out/observed7_trackcraft_dense.npz" \
    --num_frames 7 --frame_stride 1 > "$out/trackcraft.log" 2>&1
fi
echo "TRACKCRAFT_COMPLETE"

cd "$mf"
"$python" pointodyssey_interface_preflight_remote.py \
  --repo "$root/motionforesight_src" \
  --checkpoint "$checkpoint" --base-checkpoint "$base_checkpoint" \
  --user-npz "$out/observed7_user.npz" \
  --trackcraft-dense "$out/observed7_trackcraft_dense.npz" \
  --query-json "$out/query_manifest.json" \
  --pseudo-gt-bundle "$out/pointodyssey_gt_bundle.npz" \
  --cache-npz "$out/p0_h_cache_aligned.npz" \
  --output-json "$out/formal_interface_aligned.json" \
  --ground-truth-source pointodyssey_v1.2_exact_rendered_tracks \
  --query-count 48 --fps 15 > "$out/motionforesight.log" 2>&1
echo "MOTIONFORESIGHT_COMPLETE"
