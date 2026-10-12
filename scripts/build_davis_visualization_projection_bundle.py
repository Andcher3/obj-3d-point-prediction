#!/usr/bin/env python3
"""Project existing DAVIS TrackCraft3R query trajectories for visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    return np.einsum("tij,tnj->tni", matrix, xyz1)[..., :3]


def project(points: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = transform(w2c, points)
    fx, fy, cx, cy = intrinsics
    valid = np.isfinite(camera).all(-1) & (camera[..., 2] > 1e-8)
    uv = np.full(camera.shape[:-1] + (2,), np.nan, dtype=np.float64)
    z = np.maximum(camera[..., 2], 1e-8)
    uv[..., 0] = fx * camera[..., 0] / z + cx
    uv[..., 1] = fy * camera[..., 1] / z + cy
    return uv, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.video_dir
    full_user = np.load(root / "full22_user.npz", allow_pickle=True)
    observed_user = np.load(root / "observed7_user.npz", allow_pickle=True)
    observed_dense = np.load(root / "observed7_trackcraft_dense.npz")
    full_query = np.load(root / "full22_query_tracks.npz", allow_pickle=True)
    manifest = json.loads((root / "query_t0_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "query_frame0_mapping.json").read_text(encoding="utf-8"))

    full_points = np.asarray(full_query["trackcraft3r_tracks"], dtype=np.float64)
    if full_points.shape != (22, 48, 3):
        raise ValueError(f"unexpected full trajectory shape: {full_points.shape}")
    dense = np.asarray(observed_dense["track_map"], dtype=np.float64)
    dense_uv = np.asarray(mapping["query_uv_frame0_dense"], dtype=np.int64)
    observed_points = dense[:, dense_uv[:, 1], dense_uv[:, 0]]
    if observed_points.shape != (7, 48, 3):
        raise ValueError(f"unexpected observed trajectory shape: {observed_points.shape}")

    full_w2c = np.asarray(full_user["extrinsics_w2c"], dtype=np.float64)
    observed_w2c = np.asarray(observed_user["extrinsics_w2c"], dtype=np.float64)
    full_intrinsics = np.asarray(full_user["fx_fy_cx_cy"], dtype=np.float64)
    observed_intrinsics = np.asarray(observed_user["fx_fy_cx_cy"], dtype=np.float64)
    full_uv, full_valid = project(full_points, full_w2c, full_intrinsics)
    observed_uv, observed_valid = project(observed_points, observed_w2c, observed_intrinsics)

    output = {
        "video_id": manifest["video_id"],
        "source_frame_indices": np.asarray(manifest["source_frame_indices"], dtype=np.int32),
        "observed_source_frame_indices": np.asarray(manifest["observed_source_frame_indices"], dtype=np.int32),
        "t0_dense_frame_index": np.asarray(manifest["t0_dense_frame_index"], dtype=np.int32),
        "query_uv_t0_processed": np.asarray(manifest["query_sets"]["48"], dtype=np.float32),
        "query_uv_frame0_dense": dense_uv.astype(np.float32),
        "teacher_points_frame0_camera": full_points.astype(np.float32),
        "observed_points_frame0_camera": observed_points.astype(np.float32),
        "teacher_uv_processed": full_uv.astype(np.float32),
        "observed_uv_processed": observed_uv.astype(np.float32),
        "teacher_valid": full_valid,
        "observed_valid": observed_valid,
        "processed_resolution_hw": np.asarray(full_user["depth_map"].shape[1:3], dtype=np.int32),
        "time_basis": np.asarray("consecutive_davis_source_frame_index"),
        "preview_mp4_fps_used": np.asarray(False),
        "teacher_source": np.asarray("existing full22_query_tracks.npz; offline full-video TrackCraft3R query-only"),
        "observed_source": np.asarray("existing observed7_trackcraft_dense.npz sampled by query_frame0_mapping.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps({
        "video_id": manifest["video_id"],
        "output": str(args.output),
        "teacher_shape": list(full_points.shape),
        "observed_shape": list(observed_points.shape),
        "teacher_finite_fraction": float(full_valid.mean()),
        "observed_finite_fraction": float(observed_valid.mean()),
        "processed_resolution_hw": [int(x) for x in output["processed_resolution_hw"]],
    }, indent=2))


if __name__ == "__main__":
    main()
