#!/usr/bin/env python3
"""Prepare an exact 7-frame, 15 fps DA3 input for MotionForesight E3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs=7, required=True)
    parser.add_argument("--clip-start-seconds", type=float, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-name", default="depth-anything/DA3-BASE")
    parser.add_argument("--process-res", type=int, default=518)
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    indices = np.asarray(args.frame_indices, dtype=np.int32)
    if not np.all(np.diff(indices) == 1):
        raise ValueError("the seven observed frames must be consecutive")
    frames = [np.load(args.frames_dir / f"{index:04d}.npy") for index in indices]
    model = DepthAnything3.from_pretrained(args.model_name).to("cuda").eval()
    with torch.no_grad():
        prediction = model.inference(frames, process_res=args.process_res)

    rgb = np.asarray(prediction.processed_images, dtype=np.uint8)
    depth = np.asarray(prediction.depth, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    count = len(indices)
    w2c = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    w2c[:, :3, :4] = extrinsics
    first_c2w = np.linalg.inv(w2c[0])
    w2c = np.stack([pose @ first_c2w for pose in w2c]).astype(np.float32)

    height, width = depth.shape[1:]
    source_height, source_width = rgb.shape[1:3]
    if (source_height, source_width) != (height, width):
        rgb = np.stack([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in rgb
        ])
    k = intrinsics[0].copy()
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    jpeg = np.asarray([
        cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )[1].tobytes()
        for frame in rgb
    ], dtype=object)
    timestamps = args.clip_start_seconds + indices.astype(np.float64) / args.fps

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        images_jpeg_bytes=jpeg,
        rgb=rgb,
        depth_map=depth,
        extrinsics_w2c=w2c,
        fx_fy_cx_cy=np.asarray([fx, fy, cx, cy], dtype=np.float64),
        frame_indices=indices,
        timestamps_s=timestamps,
    )
    payload = {
        "status": "PASS",
        "scope": "observed-only formal E3 input",
        "frame_indices": indices.tolist(),
        "timestamps_s": timestamps.tolist(),
        "fps": args.fps,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "extrinsics_shape": list(w2c.shape),
        "coordinate_origin": "first observed camera",
        "future_frames_supplied": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
