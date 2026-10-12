#!/usr/bin/env python3
"""Build a bottle input beginning at raw frame 12, after visible motion starts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/projects/molmo-raw-test/preprocessed")
START = 12

source = np.load(ROOT / "test_video_user.npz", allow_pickle=True)
payload = {key: source[key][START:] if key in ("images_jpeg_bytes", "depth_map", "extrinsics_w2c") else source[key]
           for key in source.files}
np.savez_compressed(ROOT / "test_video_user_start12.npz", **payload)

extrinsics = np.load(ROOT / "da3/extrinsics.npy")
intrinsics = np.load(ROOT / "da3/intrinsics.npy")
indices = np.arange(START, START + 24, 2, dtype=np.int32)
scale_x, scale_y = 832.0 / 1280.0, 480.0 / 720.0
k0 = intrinsics[indices[0]]
fx_fy_cx_cy = np.asarray(
    [k0[0, 0] * scale_x, k0[1, 1] * scale_y, k0[0, 2] * scale_x, k0[1, 2] * scale_y],
    dtype=np.float32,
)
np.savez_compressed(
    ROOT / "test_video_camera_start12_stride2.npz",
    extrinsics_w2c=extrinsics[indices].astype(np.float32),
    frame_indices=indices,
    fx_fy_cx_cy=fx_fy_cx_cy,
)

# Propagate the original bottle points to raw frame 12 with the first stride-2 run,
# then project them into the new reference image.
dense = np.load(ROOT / "test_video_dense_stride2.npz")
queries = json.loads((ROOT / "test_video_queries.json").read_text(encoding="utf-8"))
uv0 = np.asarray(queries["frame0_query_pixels"], np.int64)
world_at_start = dense["track_map"][6, uv0[:, 1], uv0[:, 0]]
homogeneous = np.concatenate([world_at_start, np.ones((len(world_at_start), 1), np.float32)], axis=1)
camera_xyz = (extrinsics[START].astype(np.float32) @ homogeneous.T).T[:, :3]
k = np.asarray([[fx_fy_cx_cy[0], 0, fx_fy_cx_cy[2]],
                [0, fx_fy_cx_cy[1], fx_fy_cx_cy[3]], [0, 0, 1]], np.float32)
uv = np.stack([k[0, 0] * camera_xyz[:, 0] / camera_xyz[:, 2] + k[0, 2],
               k[1, 1] * camera_xyz[:, 1] / camera_xyz[:, 2] + k[1, 2]], axis=1)
uv = np.rint(uv).astype(int)
(ROOT / "test_video_queries_start12.json").write_text(
    json.dumps({"frame0_query_pixels": uv.tolist(), "source_raw_frame": START}, indent=2), encoding="utf-8"
)
print("indices", indices.tolist())
print("queries", uv.tolist())
