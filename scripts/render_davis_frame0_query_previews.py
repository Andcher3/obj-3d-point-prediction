#!/usr/bin/env python3
"""Render corrected frame-0 query previews for the DAVIS object probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def normalized_bbox(points: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.stack(
        [
            (points[:, 0] - x0) / max(x1 - x0, 1.0),
            (points[:, 1] - y0) / max(y1 - y0, 1.0),
        ],
        axis=1,
    )


def deterministic_fps(mask: np.ndarray, count: int, min_distance: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < count:
        raise ValueError(f"only {len(xs)} candidates for {count} queries")
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    norm = normalized_bbox(points, bbox)
    centroid = norm.mean(0)
    first_order = np.lexsort(
        (points[:, 0], points[:, 1], np.linalg.norm(norm - centroid, axis=1))
    )
    chosen = [int(first_order[0])]
    while len(chosen) < count:
        distance = np.linalg.norm(
            points[:, None] - points[np.asarray(chosen)][None], axis=-1
        ).min(1)
        distance[np.asarray(chosen)] = -np.inf
        valid = np.flatnonzero(distance >= min_distance)
        pool = valid if len(valid) else np.flatnonzero(np.isfinite(distance))
        maximum = float(distance[pool].max())
        ties = pool[np.isclose(distance[pool], maximum, rtol=0.0, atol=1e-9)]
        chosen.append(
            int(sorted(ties.tolist(), key=lambda i: (points[i, 1], points[i, 0]))[0])
        )
    return points[np.asarray(chosen)].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--frame-index", type=int, default=14)
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--masks-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--process-res", type=int, default=392)
    ap.add_argument("--query-count", type=int, default=48)
    ap.add_argument("--logical-instance-id", type=int, default=1)
    args = ap.parse_args()

    image_path = args.frames_root / args.sequence / f"{args.frame_index:05d}.jpg"
    mask_path = args.masks_root / args.sequence / f"{args.frame_index:05d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    source_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    target_source = source_mask == args.logical_instance_id
    source_h, source_w = target_source.shape
    process_h = int(round(args.process_res * source_h / source_w))
    target = cv2.resize(
        target_source.astype(np.uint8),
        (args.process_res, process_h),
        interpolation=cv2.INTER_NEAREST,
    )
    target = cv2.erode(target, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    ys, xs = np.nonzero(target)
    if len(xs) < args.query_count:
        raise ValueError(f"too few target pixels after erosion: {len(xs)}")
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    bbox_diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    min_distance = max(4.0, 0.02 * bbox_diag)
    queries_processed = deterministic_fps(target, args.query_count, min_distance)
    queries_source = queries_processed * np.asarray(
        [source_w / args.process_res, source_h / process_h], dtype=np.float32
    )

    overlay = image.copy()
    mask_u8 = target_source.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2, cv2.LINE_AA)
    shade = np.zeros_like(overlay)
    shade[target_source] = (0, 80, 0)
    overlay = cv2.addWeighted(overlay, 1.0, shade, 0.12, 0)
    for idx, (x, y) in enumerate(queries_source):
        p = (int(round(float(x))), int(round(float(y))))
        cv2.circle(overlay, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(idx),
            (p[0] + 5, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    banner = (
        f"{args.sequence} | corrected query frame: source {args.frame_index} "
        f"(model frame 0) | {args.query_count} queries"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        banner,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    print(
        {
            "sequence": args.sequence,
            "frame_index": args.frame_index,
            "query_count": args.query_count,
            "source_resolution_hw": [source_h, source_w],
            "processed_resolution_hw": [process_h, args.process_res],
            "candidate_pixels": int(target.sum()),
            "min_query_distance_processed_px": min_distance,
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
