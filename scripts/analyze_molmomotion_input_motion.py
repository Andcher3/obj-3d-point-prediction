#!/usr/bin/env python3
"""Compare temporal and geometric statistics of MolmoMotion tracker inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transform_points(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((*xyz.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate([xyz, ones], axis=-1)
    return np.einsum("ij,...j->...i", matrix.astype(np.float32), homogeneous)[..., :3]


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[..., 2], 1e-6, None)
    return np.stack(
        [k[0, 0] * xyz[..., 0] / z + k[0, 2], k[1, 1] * xyz[..., 1] / z + k[1, 2]],
        axis=-1,
    )


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None] - points[None]
    return np.linalg.norm(delta, axis=-1)


def analyse(spec: dict, history_size: int = 3) -> dict:
    camera = np.load(spec["camera_npz"], allow_pickle=True)
    dense = np.load(spec["dense_npz"])
    track_map = np.asarray(dense["track_map"], np.float32)
    frame_indices = np.asarray(camera["frame_indices"], np.int32)
    extrinsics_key = "extrinsics_w2c" if "extrinsics_w2c" in camera.files else "extrinsics"
    extrinsics = np.asarray(camera[extrinsics_key], np.float32)
    fx, fy, cx, cy = np.asarray(camera["fx_fy_cx_cy"], np.float32)
    k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    query_payload = json.loads(Path(spec["query_json"]).read_text(encoding="utf-8"))
    query_key = "frame0_query_pixels" if "frame0_query_pixels" in query_payload else "query_points"
    query_uv0 = np.asarray(query_payload[query_key], np.int64)

    u0, v0 = query_uv0[:, 0], query_uv0[:, 1]
    xyz_world = np.stack([track_map[t, v0, u0] for t in range(track_map.shape[0])], axis=0)
    t0 = history_size - 1
    xyz = transform_points(extrinsics[t0], xyz_world)
    uv = project(xyz, k)
    history = xyz[:history_size]
    history_uv = uv[:history_size]

    point_steps = np.linalg.norm(history[1:] - history[:-1], axis=-1)
    centroid = history.mean(axis=1)
    centroid_steps = np.linalg.norm(centroid[1:] - centroid[:-1], axis=-1)
    pixel_steps = np.linalg.norm(history_uv[1:] - history_uv[:-1], axis=-1)
    pairwise = np.stack([pairwise_distances(frame) for frame in history], axis=0)
    pairwise_change = np.abs(pairwise[1:] - pairwise[:-1])
    future_steps = np.linalg.norm(xyz[history_size:] - xyz[history_size - 1:-1], axis=-1)
    all_point_steps = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    all_centroid = xyz.mean(axis=1)
    all_centroid_steps = np.linalg.norm(all_centroid[1:] - all_centroid[:-1], axis=-1)
    all_pixel_steps = np.linalg.norm(uv[1:] - uv[:-1], axis=-1)

    stride = float(np.median(np.diff(frame_indices))) if len(frame_indices) > 1 else float("nan")
    return {
        "sample_id": spec["id"],
        "frame_indices": frame_indices.tolist(),
        "source_fps": float(spec["source_fps"]),
        "raw_frame_stride": stride,
        "effective_sampling_fps": float(spec["source_fps"] / stride),
        "history_duration_s": float((frame_indices[t0] - frame_indices[0]) / spec["source_fps"]),
        "query_count": int(len(query_uv0)),
        "query_bbox_xyxy": [
            int(query_uv0[:, 0].min()), int(query_uv0[:, 1].min()),
            int(query_uv0[:, 0].max()), int(query_uv0[:, 1].max()),
        ],
        "depth_mean_m": float(history[..., 2].mean()),
        "depth_range_m": [float(history[..., 2].min()), float(history[..., 2].max())],
        "history_point_step_mean_m": point_steps.mean(axis=1).tolist(),
        "history_point_step_max_m": point_steps.max(axis=1).tolist(),
        "history_centroid_step_m": centroid_steps.tolist(),
        "history_pixel_step_mean_px": pixel_steps.mean(axis=1).tolist(),
        "history_pixel_step_max_px": pixel_steps.max(axis=1).tolist(),
        "history_pairwise_deformation_mean_m": pairwise_change.mean(axis=(1, 2)).tolist(),
        "history_pairwise_deformation_max_m": pairwise_change.max(axis=(1, 2)).tolist(),
        "object_pairwise_diameter_at_t0_m": float(pairwise[t0].max()),
        "offline_future_point_path_mean_m": float(future_steps.sum(axis=0).mean()),
        "offline_future_centroid_delta_m": (xyz[-1].mean(axis=0) - history[-1].mean(axis=0)).tolist(),
        "all_point_step_mean_m": all_point_steps.mean(axis=1).tolist(),
        "all_centroid_step_m": all_centroid_steps.tolist(),
        "all_pixel_step_mean_px": all_pixel_steps.mean(axis=1).tolist(),
        "finite": bool(np.isfinite(xyz).all() and np.isfinite(uv).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.spec_json.read_text(encoding="utf-8"))["samples"]
    result = {"samples": [analyse(spec) for spec in specs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
