#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/molmo-motion
export PYTHONPATH=/root/autodl-tmp/projects/TrajLoom-probe/pytorch-image-models:/root/autodl-tmp/projects/TrajLoom-probe/Wan2.1:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=/root/autodl-tmp/projects/TrajLoom-probe
cd "$ROOT"
GEN_CONFIG=configs/trajloom_generator_probe.json
GEN_CKPT=checkpoints/models/trajloom_generator.pt
VIS_CONFIG=TrajLoom/configs/vis_predictor_config.json
VIS_CKPT=checkpoints/models/trajloom_visibility.pt

run_case() {
  local stem="$1" label="$2" seed="$3" caption="$4"
  local out="outputs/text_matrix/${stem}/${label}/seed_${seed}"
  mkdir -p "$out"
  if [[ -f "$out/${stem}/pred_tracks.npy" && -f "$out/${stem}/meta.json" ]]; then
    echo "[skip] ${stem} ${label} seed=${seed}"
    return
  fi
  echo "[run] ${stem} ${label} seed=${seed} caption=${caption}"
  local text_args=(--caption "$caption")
  if [[ "$label" == "none" ]]; then
    text_args=(--no_text_cond --caption "")
  fi
  python TrajLoom/run_trajloom_generator.py \
    --gen_config "$GEN_CONFIG" \
    --gen_ckpt "$GEN_CKPT" \
    --vis_config "$VIS_CONFIG" \
    --vis_ckpt "$VIS_CKPT" \
    --video_path "inputs/videos/${stem}.mp4" \
    --gt_dir inputs/tracks \
    --out_dir "$out" \
    --pred_len 81 \
    --noise_seed "$seed" \
    --steps 20 \
    "${text_args[@]}" \
    --device cuda:0 \
    --dtype fp16
}

# Same visual input, fixed noise seed, four semantic conditions.
run_case 10790317943435558913 none 42 ""
run_case 10790317943435558913 consistent 42 "The robot moves the colored board forward across the table."
run_case 10790317943435558913 static 42 "The robot keeps the colored board still on the table."
run_case 10790317943435558913 conflict 42 "The robot moves the colored board backward away from the camera."

run_case gear_v2_9-wrist_front_rgb_img-0 none 42 ""
run_case gear_v2_9-wrist_front_rgb_img-0 consistent 42 "The robot moves the green object forward along the workspace."
run_case gear_v2_9-wrist_front_rgb_img-0 static 42 "The robot keeps the green object still in its current position."
run_case gear_v2_9-wrist_front_rgb_img-0 conflict 42 "The robot moves the green object backward along the workspace."

# Repeated noise draws for the scene-consistent caption.
run_case 10790317943435558913 consistent 43 "The robot moves the colored board forward across the table."
run_case 10790317943435558913 consistent 44 "The robot moves the colored board forward across the table."
run_case gear_v2_9-wrist_front_rgb_img-0 consistent 43 "The robot moves the green object forward along the workspace."
run_case gear_v2_9-wrist_front_rgb_img-0 consistent 44 "The robot moves the green object forward along the workspace."

echo "TEXT_MATRIX_DONE"
