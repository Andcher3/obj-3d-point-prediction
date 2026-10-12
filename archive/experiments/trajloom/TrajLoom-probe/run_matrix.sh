#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/molmo-motion
export PYTHONPATH=/root/autodl-tmp/projects/TrajLoom-probe/pytorch-image-models:/root/autodl-tmp/projects/TrajLoom-probe/Wan2.1:${PYTHONPATH:-}
for stem in 10790317943435558913 gear_v2_9-wrist_front_rgb_img-0; do
  for seed in 42 43 44; do
    out="outputs/matrix/${stem}/seed_${seed}"
    rm -rf "$out"
    python TrajLoom/run_trajloom_generator.py \
      --gen_config configs/trajloom_generator_probe.json \
      --gen_ckpt checkpoints/models/trajloom_generator.pt \
      --vis_config TrajLoom/configs/vis_predictor_config.json \
      --vis_ckpt checkpoints/models/trajloom_visibility.pt \
      --video_path "inputs/videos/${stem}.mp4" \
      --gt_dir inputs/tracks --out_dir "$out" \
      --pred_len 81 --noise_seed "$seed" --steps 20 \
      --no_text_cond --device cuda:0 --dtype fp16
  done
done
