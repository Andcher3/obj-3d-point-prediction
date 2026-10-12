#!/usr/bin/env python3
"""Deterministically sample 48 actual frame-0 object pixels for DAVIS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample(mask: np.ndarray, count: int, min_px: float) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if len(xx) < count:
        raise ValueError(f"only {len(xx)} frame-0 candidates for {count} queries")
    pts = np.stack([xx, yy], -1).astype(np.float64)
    span = np.maximum(np.ptp(pts, axis=0), 1.0)
    normalized = (pts - pts.min(0)) / span
    center = normalized.mean(0)
    closest = np.linalg.norm(normalized - center, axis=1)
    first = int(np.lexsort((xx, yy, closest))[0])
    chosen = [first]
    norm_distance = np.linalg.norm(normalized - normalized[first], axis=1)
    pixel_distance = np.linalg.norm(pts - pts[first], axis=1)
    while len(chosen) < count:
        candidates = np.flatnonzero(pixel_distance >= min_px)
        if len(candidates) == 0:
            raise ValueError(f"FPS minimum spacing {min_px:.3f}px only permits {len(chosen)} queries")
        best = np.max(norm_distance[candidates])
        ties = candidates[np.isclose(norm_distance[candidates], best, rtol=0, atol=1e-12)]
        index = int(ties[np.lexsort((xx[ties], yy[ties]))][0])
        chosen.append(index)
        norm_distance = np.minimum(norm_distance, np.linalg.norm(normalized - normalized[index], axis=1))
        pixel_distance = np.minimum(pixel_distance, np.linalg.norm(pts - pts[index], axis=1))
    return pts[chosen].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--observed-user", type=Path, required=True)
    ap.add_argument("--full-user", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.observed_user, allow_pickle=True) as obs, np.load(args.full_user, allow_pickle=True) as full:
        obs_indices = np.asarray(obs["frame_indices"]).tolist()
        full_indices = np.asarray(full["frame_indices"]).tolist()
        if obs_indices != list(range(14, 21)) or full_indices != list(range(14, 36)):
            raise ValueError(f"unexpected source frame indices: {obs_indices} / {full_indices}")
        depth = np.asarray(obs["depth_map"][0], dtype=np.float64)
        full_depth = np.asarray(full["depth_map"][0], dtype=np.float64)
        if depth.shape != full_depth.shape:
            raise ValueError("observed/full DA3 resolution mismatch")
    indexed = np.asarray(Image.open(args.mask), dtype=np.uint8)
    if indexed.ndim != 2 or not np.any(indexed == 1):
        raise ValueError("frame-0 palette index 1 mask missing")
    h, w = depth.shape
    mask = cv2.resize((indexed == 1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    eroded = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(full_depth) & (full_depth > 0)
    candidates = eroded & valid
    yy, xx = np.nonzero(eroded)
    if not len(xx):
        raise ValueError("frame-0 eroded mask empty")
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    diagonal = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    minimum = max(4.0, 0.02 * diagonal)
    query = sample(candidates, 48, minimum)
    src_h, src_w = indexed.shape
    query_src = query * np.asarray([src_w / w, src_h / h], dtype=np.float32)
    payload = {
        "video_id": args.video_id,
        "protocol": "frame0_direct_query_2026-09-28",
        "source_dataset": "DAVIS-2017-trainval-480p",
        "logical_palette_instance_id": 1,
        "query_reference_source_frame_index": 14,
        "query_reference_model_frame_index": 0,
        "output_camera_source_frame_index": 20,
        "output_camera_model_frame_index": 6,
        "source_frame_indices": full_indices,
        "observed_source_frame_indices": obs_indices,
        "evaluated_future_source_frame_indices": list(range(21, 30)),
        "time_basis": "consecutive DAVIS source frame indices; capture fps unknown",
        "preview_mp4_fps_used": False,
        "mask_shape_hw": [h, w],
        "source_mask_shape_hw": [src_h, src_w],
        "bbox_xyxy": bbox,
        "bbox_diagonal_px": diagonal,
        "erosion_pixels_at_processed_resolution": 2,
        "candidate_pixel_count": int(candidates.sum()),
        "min_query_distance_px": minimum,
        "query_sets": {"48": query.tolist()},
        "query_uv_frame0_source_image": query_src.tolist(),
        "query_selection": "normalized-bbox deterministic FPS from source frame 14 palette mask and positive finite observed/full frame-0 DA3 depths",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video_id": args.video_id, "query_count": 48, "candidate_pixel_count": int(candidates.sum()), "minimum_px": minimum}, indent=2))


if __name__ == "__main__":
    main()
