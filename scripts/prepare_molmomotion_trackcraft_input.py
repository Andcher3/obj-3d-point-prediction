#!/usr/bin/env python3
"""Align DA3 camera data and bottle query points to TrackCraft3R's 832x480 output."""
from pathlib import Path
import json
import numpy as np

ROOT = Path("/root/autodl-tmp/projects/molmo-raw-test/preprocessed")
depth = np.load(ROOT / "da3/depth.npy")
extr = np.load(ROOT / "da3/extrinsics.npy")
intr = np.load(ROOT / "da3/intrinsics.npy")
indices = np.arange(0, 48, 4, dtype=np.int32)

sx, sy = 832.0 / 1280.0, 480.0 / 720.0
k0 = intr[indices[0]].copy()
fx_fy_cx_cy = np.array(
    [k0[0, 0] * sx, k0[1, 1] * sy, k0[0, 2] * sx, k0[1, 2] * sy],
    dtype=np.float32,
)
np.savez_compressed(
    ROOT / "test_video_camera_aligned.npz",
    extrinsics_w2c=extr[indices].astype(np.float32),
    frame_indices=indices,
    fx_fy_cx_cy=fx_fy_cx_cy,
)

# Eight points placed on visible parts of the bottle in raw frame 0.
uv_raw = np.array(
    [[455, 350], [500, 350], [430, 390], [525, 390],
     [445, 450], [525, 450], [455, 530], [520, 535]],
    dtype=np.float32,
)
uv_dense = np.rint(uv_raw * np.array([sx, sy], np.float32)).astype(int)
(ROOT / "test_video_queries.json").write_text(
    json.dumps({"frame0_query_pixels": uv_dense.tolist(),
                "frame0_query_pixels_raw_1280x720": uv_raw.astype(int).tolist()}, indent=2),
    encoding="utf-8",
)
print("indices", indices.tolist())
print("fx_fy_cx_cy", fx_fy_cx_cy.tolist())
print("queries", uv_dense.tolist())
