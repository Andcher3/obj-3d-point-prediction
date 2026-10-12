#!/usr/bin/env python3
"""Prepare formal E2 timestamp grids, masks, and nested query manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
    from scipy import ndimage


TARGET_FPS = 15.0
HISTORY_OFFSETS = (-2.0 / TARGET_FPS, -1.0 / TARGET_FPS, 0.0)
FUTURE_OFFSETS = tuple(index / TARGET_FPS for index in range(1, 10))


def nearest_timestamp_indices(timestamps: np.ndarray, targets: np.ndarray, tolerance: float) -> tuple[list[int | None], list[float]]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    indices: list[int | None] = []
    errors: list[float] = []
    for target in np.asarray(targets, dtype=np.float64):
        index = int(np.argmin(np.abs(timestamps - target)))
        error = float(abs(timestamps[index] - target))
        indices.append(index if error <= tolerance else None)
        errors.append(error)
    return indices, errors


def polygon_mask(height: int, width: int, polygon_xy: list[list[float]], erosion_pixels: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.rint(np.asarray(polygon_xy, dtype=np.float32)).astype(np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must contain at least three [x, y] points")
    if cv2 is not None:
        cv2.fillPoly(mask, [polygon], 1)
    else:
        # Dependency-light polygon rasterization for the local preparation path.
        from matplotlib.path import Path as MplPath
        yy, xx = np.mgrid[:height, :width]
        mask[:] = MplPath(polygon.astype(float)).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(height, width)
    if erosion_pixels > 0:
        kernel_size = 2 * erosion_pixels + 1
        if cv2 is not None:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        else:
            mask = ndimage.binary_erosion(mask.astype(bool), structure=np.ones((kernel_size, kernel_size), dtype=bool), iterations=1)
    return mask.astype(bool)


def normalized_bbox_coordinates(points_xy: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    return np.stack([(points_xy[:, 0] - x0) / width, (points_xy[:, 1] - y0) / height], axis=1)


def deterministic_fps(
    points_xy: np.ndarray,
    candidate_mask: np.ndarray,
    count: int,
    min_distance_px: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[int]:
    ys, xs = np.nonzero(candidate_mask)
    if len(xs) == 0:
        return []
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    if bbox is None:
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    normalized = normalized_bbox_coordinates(points, bbox)
    centroid = normalized.mean(axis=0)
    distances_to_centroid = np.linalg.norm(normalized - centroid[None], axis=1)
    order = np.lexsort((points[:, 0], points[:, 1], distances_to_centroid))
    selected_local: list[int] = [int(order[0])]
    selected_points = points[selected_local]
    while len(selected_local) < min(count, len(points)):
        distance_to_selected = np.linalg.norm(points[:, None, :] - selected_points[None, :, :], axis=-1).min(axis=1)
        distance_to_selected[np.asarray(selected_local)] = -np.inf
        valid = np.where(distance_to_selected >= min_distance_px)[0]
        pool = valid if len(valid) else np.where(distance_to_selected > -np.inf)[0]
        if len(pool) == 0:
            break
        max_distance = distance_to_selected[pool].max()
        tied = pool[np.isclose(distance_to_selected[pool], max_distance, rtol=0.0, atol=1e-9)]
        chosen = int(sorted(tied.tolist(), key=lambda index: (points[index, 1], points[index, 0]))[0])
        selected_local.append(chosen)
        selected_points = points[selected_local]
    return [int(ys[index]) * candidate_mask.shape[1] + int(xs[index]) for index in selected_local]


def prepare_video(spec: dict, output_dir: Path) -> dict:
    video_id = str(spec["video_id"])
    timestamps = np.asarray(json.loads(Path(spec["timestamps_json"]).read_text(encoding="utf-8")), dtype=np.float64)
    t0 = float(spec["t0_seconds"])
    tolerance = 0.5 / TARGET_FPS
    history_targets = np.asarray([t0 + offset for offset in HISTORY_OFFSETS], dtype=np.float64)
    future_targets = np.asarray([t0 + offset for offset in FUTURE_OFFSETS], dtype=np.float64)
    history_indices, history_errors = nearest_timestamp_indices(timestamps, history_targets, tolerance)
    future_indices, future_errors = nearest_timestamp_indices(timestamps, future_targets, tolerance)

    pointmaps = np.load(spec["pointmap_history_npy"])
    if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3:
        raise ValueError("pointmap_history_npy must have shape [history, height, width, 3]")
    if len(pointmaps) < 3:
        raise ValueError("three history pointmaps are required")
    height, width = pointmaps.shape[1:3]
    mask = polygon_mask(height, width, spec["polygon_xy"], int(spec.get("erosion_pixels", 2)))
    finite_positive = np.isfinite(pointmaps[:3]).all(axis=-1) & (pointmaps[:3, ..., 2] > 0)
    candidates = mask & finite_positive.all(axis=0)
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        raise ValueError(f"{video_id}: no valid query candidates")
    polygon = np.asarray(spec["polygon_xy"], dtype=np.float64)
    bbox = [float(polygon[:, 0].min()), float(polygon[:, 1].min()), float(polygon[:, 0].max()), float(polygon[:, 1].max())]
    bbox_diagonal = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    min_distance = max(4.0, 0.02 * bbox_diagonal)
    linear_indices = deterministic_fps(candidates, candidates, 64, min_distance, bbox=tuple(bbox))
    query_xy = [[int(index % width), int(index // width)] for index in linear_indices]
    query_xyz_history = np.asarray([[pointmaps[t, y, x].tolist() for x, y in query_xy] for t in range(3)], dtype=np.float32)
    query_valid = np.isfinite(query_xyz_history).all(axis=-1) & (query_xyz_history[..., 2] > 0)
    payload = {
        "video_id": video_id,
        "source_video": spec.get("source_video"),
        "motion_category": spec.get("motion_category"),
        "target_fps": TARGET_FPS,
        "window_seconds": 2.0,
        "t0_seconds": t0,
        "history_target_seconds": history_targets.tolist(),
        "future_target_seconds": future_targets.tolist(),
        "history_source_indices": history_indices,
        "future_source_indices": future_indices,
        "history_match_errors_seconds": history_errors,
        "future_match_errors_seconds": future_errors,
        "timestamp_tolerance_seconds": tolerance,
        "timestamp_valid": all(index is not None for index in history_indices + future_indices),
        "polygon_xy": polygon.tolist(),
        "erosion_pixels": int(spec.get("erosion_pixels", 2)),
        "mask_shape_hw": [height, width],
        "mask_valid_pixel_count": int(mask.sum()),
        "candidate_pixel_count": int(candidates.sum()),
        "bbox_xyxy": bbox,
        "min_query_distance_px": min_distance,
        "query_count": len(query_xy),
        "query_xy": query_xy,
        "query_sets": {"32": query_xy[:32], "48": query_xy[:48], "64": query_xy[:64]},
        "query_xyz_history_shape": list(query_xyz_history.shape),
        "query_history_valid_fraction": float(query_valid.mean()),
        "query_history_xyz_path": f"{video_id}_query_history_xyz.npy",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / payload["query_history_xyz_path"], query_xyz_history)
    with (output_dir / f"{video_id}_query_mask.npy").open("wb") as handle:
        np.save(handle, mask.astype(np.uint8))
    (output_dir / f"{video_id}_query_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
    videos = spec["videos"] if "videos" in spec else [spec]
    manifests = [prepare_video(video, args.output_dir) for video in videos]
    summary = {
        "target_fps": TARGET_FPS,
        "history_offsets_seconds": list(HISTORY_OFFSETS),
        "future_offsets_seconds": list(FUTURE_OFFSETS),
        "videos": manifests,
        "formal_query_gate_passed": all(item["query_count"] >= 48 and item["timestamp_valid"] for item in manifests),
    }
    (args.output_dir / "formal_e2_query_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
