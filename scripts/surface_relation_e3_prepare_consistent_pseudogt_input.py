#!/usr/bin/env python3
"""Build an offline future input registered to an observed-only DA3 sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-npz", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--t0-index", type=int, required=True)
    parser.add_argument("--clip-start-seconds", type=float, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--model-name", default="depth-anything/DA3-BASE")
    parser.add_argument("--process-res", type=int, default=518)
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    observed = np.load(args.observed_npz, allow_pickle=True)
    query = json.loads(args.query_json.read_text(encoding="utf-8"))
    observed_rgb = np.asarray(observed["rgb"], dtype=np.uint8)
    observed_depth = np.asarray(observed["depth_map"], dtype=np.float32)
    observed_w2c = np.asarray(observed["extrinsics_w2c"], dtype=np.float64)
    intrinsics = np.asarray(observed["fx_fy_cx_cy"], dtype=np.float64)
    observed_indices = np.asarray(observed["frame_indices"], dtype=np.int32)
    if observed_indices[-1] != args.t0_index:
        raise ValueError("observed input does not end at the requested t0 index")

    future_indices = np.arange(args.t0_index, args.t0_index + 10, dtype=np.int32)
    future_frames = [np.load(args.frames_dir / f"{index:04d}.npy") for index in future_indices]
    model = DepthAnything3.from_pretrained(args.model_name).to("cuda").eval()
    with torch.no_grad():
        prediction = model.inference(future_frames, process_res=args.process_res)
    future_rgb = np.asarray(prediction.processed_images, dtype=np.uint8)
    future_depth = np.asarray(prediction.depth, dtype=np.float32)
    future_extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    future_w2c = np.tile(np.eye(4, dtype=np.float64), (len(future_indices), 1, 1))
    future_w2c[:, :3, :4] = future_extrinsics
    future_first_c2w = np.linalg.inv(future_w2c[0])
    future_w2c = np.stack([pose @ future_first_c2w for pose in future_w2c])

    height, width = observed_depth.shape[1:]
    if future_depth.shape[1:] != (height, width):
        future_depth = np.stack([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in future_depth
        ])
        future_rgb = np.stack([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in future_rgb
        ])
    query_uv = np.asarray(query["query_sets"]["48"], dtype=np.int64)
    source_h, source_w = map(int, query["mask_shape_hw"])
    query_uv[:, 0] = np.clip(np.rint(query_uv[:, 0] * width / source_w), 0, width - 1).astype(np.int64)
    query_uv[:, 1] = np.clip(np.rint(query_uv[:, 1] * height / source_h), 0, height - 1).astype(np.int64)
    observed_t0_depth = observed_depth[-1, query_uv[:, 1], query_uv[:, 0]]
    future_t0_depth = future_depth[0, query_uv[:, 1], query_uv[:, 0]]
    valid = (
        np.isfinite(observed_t0_depth)
        & np.isfinite(future_t0_depth)
        & (observed_t0_depth > 1e-8)
        & (future_t0_depth > 1e-8)
    )
    ratios = observed_t0_depth[valid] / future_t0_depth[valid]
    if len(ratios) < 32:
        raise RuntimeError(f"only {len(ratios)} valid t0 query depths for scale registration")
    scale = float(np.median(ratios))
    future_depth *= scale

    observed_last = observed_w2c[-1]
    combined_future_w2c = []
    for pose in future_w2c[1:]:
        combined = np.eye(4, dtype=np.float64)
        combined[:3, :3] = pose[:3, :3] @ observed_last[:3, :3]
        combined[:3, 3] = pose[:3, :3] @ observed_last[:3, 3] + scale * pose[:3, 3]
        combined_future_w2c.append(combined)
    combined_w2c = np.concatenate([
        observed_w2c,
        np.asarray(combined_future_w2c, dtype=np.float64),
    ]).astype(np.float32)
    combined_rgb = np.concatenate([observed_rgb, future_rgb[1:]], axis=0)
    combined_depth = np.concatenate([observed_depth, future_depth[1:]], axis=0)
    combined_indices = np.concatenate([observed_indices, future_indices[1:]])
    timestamps = args.clip_start_seconds + combined_indices.astype(np.float64) / args.fps
    jpeg = np.asarray([
        cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )[1].tobytes()
        for frame in combined_rgb
    ], dtype=object)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        images_jpeg_bytes=jpeg,
        rgb=combined_rgb,
        depth_map=combined_depth,
        extrinsics_w2c=combined_w2c,
        fx_fy_cx_cy=intrinsics,
        frame_indices=combined_indices,
        timestamps_s=timestamps,
        observed_frame_count=np.asarray(7, dtype=np.int32),
    )
    payload = {
        "status": "PASS",
        "scope": "offline E3 pseudo-GT input; only first seven frames are prediction conditions",
        "shape": list(combined_rgb.shape),
        "frame_indices": combined_indices.tolist(),
        "timestamps_s": timestamps.tolist(),
        "observed_frame_count": 7,
        "future_frame_count": 9,
        "t0_depth_scale": scale,
        "t0_depth_ratio_iqr": [float(np.quantile(ratios, 0.25)), float(np.quantile(ratios, 0.75))],
        "scale_estimation_source": "48 shared t0 query depths only",
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
