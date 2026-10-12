#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/molmo-motion
export PYTHONPATH=/root/autodl-tmp/projects/TrajLoom-probe/pytorch-image-models:/root/autodl-tmp/projects/TrajLoom-probe/Wan2.1:${PYTHONPATH:-}

run_case() {
  local stem="$1" key="$2" seed="$3" caption="$4"
  local out="outputs/event_text_probe/${stem}/${key}_seed${seed}"
  rm -rf "$out"
  echo "[case] stem=$stem key=$key seed=$seed caption=$caption"
  python TrajLoom/run_trajloom_generator.py \
    --gen_config configs/trajloom_generator_probe.json \
    --gen_ckpt checkpoints/models/trajloom_generator.pt \
    --vis_config TrajLoom/configs/vis_predictor_config.json \
    --vis_ckpt checkpoints/models/trajloom_visibility.pt \
    --video_path "inputs/videos/${stem}.mp4" \
    --gt_dir inputs/tracks \
    --out_dir "$out" \
    --pred_len 81 \
    --noise_seed "$seed" \
    --steps 20 \
    --caption "$caption" \
    --device cuda:0 \
    --dtype fp16
}

shape_stem="10790317943435558913"
shape_correct="The robot moves forward toward the shape-sorting board, so the board and colored blocks move downward and become larger in the camera view."
shape_opposite="The robot moves backward away from the shape-sorting board, so the board and colored blocks move upward and become smaller in the camera view."

gear_stem="gear_v2_9-wrist_front_rgb_img-0"
gear_correct="The robot moves forward toward the green cylindrical object, so the green object becomes larger and moves toward the center of the camera view."
gear_opposite="The robot moves backward away from the green cylindrical object, so the green object becomes smaller and moves away from the center of the camera view."

for seed in 42 43 44; do
  run_case "$shape_stem" correct "$seed" "$shape_correct"
  run_case "$gear_stem" correct "$seed" "$gear_correct"
done

run_case "$shape_stem" opposite 42 "$shape_opposite"
run_case "$gear_stem" opposite 42 "$gear_opposite"

