#!/usr/bin/env python3
"""Create the 15-fps camera bundle for the bottle TrackCraft stride-2 run."""

from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/projects/molmo-raw-test/preprocessed")
extrinsics = np.load(ROOT / "da3/extrinsics.npy")
intrinsics = np.load(ROOT / "da3/intrinsics.npy")
indices = np.arange(0, 24, 2, dtype=np.int32)

scale_x, scale_y = 832.0 / 1280.0, 480.0 / 720.0
k0 = intrinsics[indices[0]]
fx_fy_cx_cy = np.asarray(
    [k0[0, 0] * scale_x, k0[1, 1] * scale_y, k0[0, 2] * scale_x, k0[1, 2] * scale_y],
    dtype=np.float32,
)
np.savez_compressed(
    ROOT / "test_video_camera_stride2.npz",
    extrinsics_w2c=extrinsics[indices].astype(np.float32),
    frame_indices=indices,
    fx_fy_cx_cy=fx_fy_cx_cy,
)
print(indices.tolist())
print(fx_fy_cx_cy.tolist())
