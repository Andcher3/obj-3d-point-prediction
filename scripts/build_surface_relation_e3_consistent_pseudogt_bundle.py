#!/usr/bin/env python3
"""Sample 48-query E3 pseudo-GT from a registered 16-frame TrackCraft run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def match_queries(
    track_map: np.ndarray,
    frame_index: int,
    w2c: np.ndarray,
    intr: np.ndarray,
    query_uv: np.ndarray,
    query_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = track_map.shape[1:3]
    source_h, source_w = query_hw
    k = np.asarray([
        [intr[0] * width / source_w, 0.0, intr[2] * width / source_w],
        [0.0, intr[1] * height / source_h, intr[3] * height / source_h],
        [0.0, 0.0, 1.0],
    ])
    xyz = transform(w2c, track_map[frame_index])
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-8)
    projected = np.full((*xyz.shape[:2], 2), np.nan, dtype=np.float64)
    projected[..., 0][valid] = k[0, 0] * xyz[..., 0][valid] / xyz[..., 2][valid] + k[0, 2]
    projected[..., 1][valid] = k[1, 1] * xyz[..., 1][valid] / xyz[..., 2][valid] + k[1, 2]
    query_model = query_uv * np.asarray([width / source_w, height / source_h])
    valid_y, valid_x = np.nonzero(valid)
    _, neighbours = cKDTree(projected[valid]).query(query_model, k=min(128, len(valid_x)))
    candidates = np.unique(np.asarray(neighbours).reshape(-1))
    costs = cdist(query_model, projected[valid][candidates])
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != len(query_model) or not np.array_equal(np.sort(rows), np.arange(len(query_model))):
        raise RuntimeError("could not assign a global one-to-one full-sequence mapping")
    selected = np.empty(len(query_model), dtype=np.int64)
    selected_distances = np.empty(len(query_model), dtype=np.float32)
    selected[rows] = candidates[columns]
    selected_distances[rows] = costs[rows, columns]
    anchors = np.stack([valid_x[selected], valid_y[selected]], axis=-1).astype(np.int64)
    return anchors, selected_distances


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
    if track_map.shape[0] != 16:
        raise ValueError(f"expected 16 frames, got {track_map.shape}")
    w2c = np.asarray(camera["extrinsics_w2c"], dtype=np.float64)
    intr = np.asarray(camera["fx_fy_cx_cy"], dtype=np.float64)
    timestamps = np.asarray(camera["timestamps_s"], dtype=np.float64)
    query_uv = np.asarray(query["query_sets"]["48"], dtype=np.float32)
    anchors, errors = match_queries(
        track_map, 6, w2c[6], intr, query_uv, tuple(map(int, query["mask_shape_hw"]))
    )
    tracks = track_map[:, anchors[:, 1], anchors[:, 0]]
    tracks_t0 = transform(w2c[6], tracks)
    output = {
        "points_3d_history": tracks_t0[4:7],
        "pseudo_gt": tracks_t0[7:16],
        "validity": np.isfinite(tracks_t0).all(axis=-1) & (tracks_t0[..., 2] > 0),
        "pseudo_gt_timestamps_s": timestamps[7:16],
        "query_uv_t0_source": query_uv,
        "query_uv_frame0_dense": anchors.astype(np.float32),
        "query_t0_match_error_px": errors,
        "trajectory_source": np.asarray("trackcraft3r_e3_registered_offline_future"),
        "coordinate_frame": np.asarray("last_observed_camera"),
        "source_video": np.asarray(query["video_id"]),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output)
    print(json.dumps({
        "output": str(args.output_npz),
        "history_shape": list(output["points_3d_history"].shape),
        "pseudo_gt_shape": list(output["pseudo_gt"].shape),
        "match_error_px_median": float(np.median(errors)),
        "match_error_px_max": float(np.max(errors)),
        "valid_fraction": float(output["validity"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
