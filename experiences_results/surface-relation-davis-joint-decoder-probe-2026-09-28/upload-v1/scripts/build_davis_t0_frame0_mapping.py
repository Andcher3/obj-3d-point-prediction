#!/usr/bin/env python3
"""Map fixed DAVIS t0 queries to unique observed TrackCraft frame-0 tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    return np.einsum("ij,...j->...i", matrix, xyz1)[..., :3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-npz", type=Path, required=True)
    ap.add_argument("--camera-npz", type=Path, required=True)
    ap.add_argument("--query-t0-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    dense = np.load(args.dense_npz)
    camera = np.load(args.camera_npz, allow_pickle=True)
    manifest = json.loads(args.query_t0_json.read_text(encoding="utf-8"))
    tracks = np.asarray(dense["track_map"], dtype=np.float64)
    if tracks.shape[0] != 7:
        raise ValueError(f"expected 7 observed frames, got {tracks.shape}")
    source_h, source_w = map(int, manifest["mask_shape_hw"])
    height, width = tracks.shape[1:3]
    query_t0 = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
    query_dense = query_t0 * np.asarray([width / source_w, height / source_h])
    w2c = np.asarray(camera["extrinsics_w2c"], dtype=np.float64)
    fx, fy, cx, cy = np.asarray(camera["fx_fy_cx_cy"], dtype=np.float64)
    k = np.asarray([[fx * width / source_w, 0, cx * width / source_w], [0, fy * height / source_h, cy * height / source_h], [0, 0, 1]], dtype=np.float64)
    t0 = transform(w2c[6], tracks[6])
    valid = np.isfinite(t0).all(-1) & (t0[..., 2] > 1e-8)
    projected = np.full((height, width, 2), np.nan, dtype=np.float64)
    projected[..., 0][valid] = k[0, 0] * t0[..., 0][valid] / t0[..., 2][valid] + k[0, 2]
    projected[..., 1][valid] = k[1, 1] * t0[..., 1][valid] / t0[..., 2][valid] + k[1, 2]
    yy, xx = np.nonzero(valid)
    tree = cKDTree(projected[valid])
    distances, neighbours = tree.query(query_dense, k=min(64, len(xx)))
    if neighbours.ndim == 1:
        neighbours, distances = neighbours[:, None], distances[:, None]
    used: set[int] = set()
    chosen = []
    chosen_distance = []
    second_gap = []
    for row, row_distance in zip(neighbours, distances):
        available = [(int(index), float(distance)) for index, distance in zip(row, row_distance) if int(index) not in used]
        if not available:
            raise RuntimeError("unique t0-to-frame0 assignment failed")
        index, distance = available[0]
        used.add(index)
        chosen.append(index)
        chosen_distance.append(distance)
        second_gap.append(float(available[1][1] - distance) if len(available) > 1 else float("inf"))
    u_dense = xx[np.asarray(chosen)]
    v_dense = yy[np.asarray(chosen)]
    frame0_input = np.stack([u_dense * source_w / width, v_dense * source_h / height], axis=-1).astype(np.float32)
    bbox_diag_dense = float(manifest["bbox_diagonal_px"]) * float(np.hypot(width, height) / np.hypot(source_w, source_h))
    error = np.asarray(chosen_distance)
    payload = {
        "video_id": manifest["video_id"],
        "query_sets": {"48": frame0_input.tolist()},
        "query_uv_t0_input": query_t0.tolist(),
        "query_uv_t0_dense": query_dense.tolist(),
        "query_uv_frame0_dense": np.stack([u_dense, v_dense], axis=-1).astype(int).tolist(),
        "input_resolution_hw": [source_h, source_w],
        "dense_resolution_hw": [height, width],
        "mapping_source": "observed-only 7-frame dense TrackCraft3R",
        "uses_future_frames": False,
        "match_error_px_dense": error.tolist(),
        "second_candidate_gap_px_dense": second_gap,
        "match_error_normalized_bbox": (error / max(bbox_diag_dense, 1e-8)).tolist(),
        "match_p90_normalized_bbox": float(np.quantile(error / max(bbox_diag_dense, 1e-8), 0.9)),
        "match_max_normalized_bbox": float(np.max(error / max(bbox_diag_dense, 1e-8))),
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("video_id", "match_p90_normalized_bbox", "match_max_normalized_bbox")}, indent=2))


if __name__ == "__main__":
    main()
