#!/usr/bin/env python3
"""Sample formal nested queries from a TrackCraft3R dense trajectory map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((*points.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate([points.astype(np.float32), ones], axis=-1)
    return np.einsum("ij,...j->...i", matrix.astype(np.float32), homogeneous)[..., :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-npz", type=Path, required=True)
    parser.add_argument("--camera-npz", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    dense = np.load(args.dense_npz)
    camera = np.load(args.camera_npz, allow_pickle=True)
    query = json.loads(args.query_json.read_text(encoding="utf-8"))
    track_map = np.asarray(dense["track_map"], dtype=np.float32)
    if track_map.shape[0] != 12 or track_map.shape[-1] != 3:
        raise ValueError(f"expected [12,H,W,3] track_map, got {track_map.shape}")

    source_h, source_w = map(int, query["mask_shape_hw"])
    output_h, output_w = track_map.shape[1:3]
    query_uv_source = np.asarray(query["query_sets"]["64"], dtype=np.float32)
    query_uv_t0_model = query_uv_source * np.asarray(
        [output_w / source_w, output_h / source_h], dtype=np.float32
    )

    extrinsics = np.asarray(camera["extrinsics" if "extrinsics" in camera.files else "extrinsics_w2c"], dtype=np.float32)
    t0_w2c = extrinsics[2]

    if "intrinsics" in camera.files:
        intr = np.asarray(camera["intrinsics"], dtype=np.float32)
    elif "fx_fy_cx_cy" in camera.files:
        intr = np.asarray(camera["fx_fy_cx_cy"], dtype=np.float32)
    else:
        raise ValueError("camera NPZ lacks intrinsics/fx_fy_cx_cy")
    if intr.ndim == 3:
        k = intr[2]
    elif intr.shape == (3, 3):
        k = intr
    elif intr.shape[-1] == 4:
        fx, fy, cx, cy = intr[2] if intr.ndim == 2 else intr
        k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError(f"unsupported intrinsics shape {intr.shape}")

    # Intrinsics accompany the source image resolution; rescale to dense/model pixels.
    k_model = k.copy()
    k_model[0] *= output_w / source_w
    k_model[1] *= output_h / source_h
    # TrackCraft's dense map is indexed by frame-0 pixels, while formal queries
    # are selected on t0 (frame 2). Match each t0 query to a unique dense track
    # by projecting all frame-0-indexed tracks at t0 into the t0 image.
    t0_dense = transform(t0_w2c, track_map[2])
    dense_z = t0_dense[..., 2]
    dense_valid = np.isfinite(t0_dense).all(axis=-1) & (dense_z > 1e-8)
    dense_projected = np.full((output_h, output_w, 2), np.nan, dtype=np.float32)
    dense_projected[..., 0][dense_valid] = (
        k_model[0, 0] * t0_dense[..., 0][dense_valid] / dense_z[dense_valid] + k_model[0, 2]
    )
    dense_projected[..., 1][dense_valid] = (
        k_model[1, 1] * t0_dense[..., 1][dense_valid] / dense_z[dense_valid] + k_model[1, 2]
    )
    valid_y, valid_x = np.nonzero(dense_valid)
    tree = cKDTree(dense_projected[dense_valid])
    # Request several neighbours to enforce identity uniqueness deterministically.
    distances, neighbours = tree.query(query_uv_t0_model, k=min(32, len(valid_x)))
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
        distances = distances[:, None]
    chosen: list[int] = []
    used: set[int] = set()
    chosen_distances: list[float] = []
    for row_indices, row_distances in zip(neighbours, distances):
        selected = next((int(index) for index in row_indices if int(index) not in used), None)
        if selected is None:
            raise RuntimeError("could not assign a unique frame-0 track to every t0 query")
        used.add(selected)
        chosen.append(selected)
        chosen_distances.append(float(row_distances[np.where(row_indices == selected)[0][0]]))
    u = valid_x[np.asarray(chosen)]
    v = valid_y[np.asarray(chosen)]
    tracks_cam0 = track_map[:, v, u]
    tracks_t0 = transform(t0_w2c, tracks_cam0)
    z = np.clip(tracks_t0[2, :, 2], 1e-8, None)
    projected = np.stack([
        k_model[0, 0] * tracks_t0[2, :, 0] / z + k_model[0, 2],
        k_model[1, 1] * tracks_t0[2, :, 1] / z + k_model[1, 2],
    ], axis=-1).astype(np.float32)

    future_timestamps = np.asarray(query["future_target_seconds"], dtype=np.float64)
    pseudo_gt = tracks_t0[3:12]
    # Deterministic, mean-centred, horizon-growing point perturbation. It is an
    # E2-G integration baseline only and must not be described as a model output.
    t0_tree = cKDTree(tracks_t0[2])
    neighbour_distances, _ = t0_tree.query(tracks_t0[2], k=2)
    spacing = float(np.median(neighbour_distances[:, 1]))
    seed = int.from_bytes(hashlib.sha256(query["video_id"].encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(len(query_uv_source), 3)).astype(np.float32)
    directions -= directions.mean(axis=0, keepdims=True)
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    horizon_scale = np.linspace(1.0 / 9.0, 1.0, 9, dtype=np.float32)[:, None, None]
    controlled_perturbation = horizon_scale * directions[None] * (0.10 * spacing)
    output = {
        "points_3d_history": tracks_t0[:3],
        "points_2d_at_t0": projected,
        "intrinsics": k_model,
        "pseudo_gt": pseudo_gt,
        "future_3d": pseudo_gt + controlled_perturbation,
        "t0_seconds": np.asarray(query["t0_seconds"], dtype=np.float64),
        "future_timestamps_s": future_timestamps,
        "pseudo_gt_timestamps_s": future_timestamps.copy(),
        "query_uv_t0_source": query_uv_source,
        "query_uv_t0_model": query_uv_t0_model,
        "query_uv_frame0_model": np.stack([u, v], axis=-1).astype(np.float32),
        "query_t0_match_error_px": np.asarray(chosen_distances, dtype=np.float32),
        "validity": np.isfinite(tracks_t0).all(axis=-1) & (tracks_t0[..., 2] > 0),
        "trajectory_source": np.asarray("trackcraft3r_pseudo_gt"),
        "prediction_source": np.asarray("controlled_perturbation_baseline"),
        "controlled_perturbation_scale_m": np.asarray(0.10 * spacing, dtype=np.float32),
        "controlled_perturbation_seed": np.asarray(seed, dtype=np.uint64),
        "controlled_perturbation_definition": np.asarray(
            "mean-centred fixed per-query random direction; magnitude ramps linearly by horizon; final scale=0.10*median_t0_nearest_neighbour_distance"
        ),
        "source_video": np.asarray(query["video_id"]),
        "motion_category": np.asarray(query.get("motion_category", "unknown")),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output)
    print(json.dumps({
        "output": str(args.output_npz),
        "history_shape": list(output["points_3d_history"].shape),
        "pseudo_gt_shape": list(output["pseudo_gt"].shape),
        "finite_fraction": float(np.isfinite(tracks_t0).mean()),
        "valid_fraction": float(output["validity"].mean()),
        "t0_match_error_px_max": float(np.max(chosen_distances)),
        "t0_match_error_px_median": float(np.median(chosen_distances)),
    }, indent=2))


if __name__ == "__main__":
    main()
