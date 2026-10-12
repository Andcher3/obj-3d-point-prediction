#!/usr/bin/env python3
"""Build observed-only and offline-teacher DA3 inputs for the DAVIS probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def normalized_bbox(points: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.stack(
        [(points[:, 0] - x0) / max(x1 - x0, 1.0), (points[:, 1] - y0) / max(y1 - y0, 1.0)],
        axis=1,
    )


def deterministic_fps(mask: np.ndarray, count: int, min_distance: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < count:
        raise ValueError(f"only {len(xs)} valid candidate pixels for {count} queries")
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    norm = normalized_bbox(points, bbox)
    centroid = norm.mean(0)
    first_order = np.lexsort((points[:, 0], points[:, 1], np.linalg.norm(norm - centroid, axis=1)))
    chosen = [int(first_order[0])]
    while len(chosen) < count:
        distance = np.linalg.norm(points[:, None] - points[np.asarray(chosen)][None], axis=-1).min(1)
        distance[np.asarray(chosen)] = -np.inf
        valid = np.flatnonzero(distance >= min_distance)
        pool = valid if len(valid) else np.flatnonzero(np.isfinite(distance))
        maximum = float(distance[pool].max())
        ties = pool[np.isclose(distance[pool], maximum, rtol=0.0, atol=1e-9)]
        chosen.append(int(sorted(ties.tolist(), key=lambda i: (points[i, 1], points[i, 0]))[0]))
    return points[np.asarray(chosen)].astype(np.float32)


def save_user_npz(path: Path, prediction, frame_indices: np.ndarray, observed_count: int) -> dict:
    rgb = np.asarray(prediction.processed_images, dtype=np.uint8)
    depth = np.asarray(prediction.depth, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    w2c = np.tile(np.eye(4, dtype=np.float64), (len(frame_indices), 1, 1))
    w2c[:, :3, :4] = extrinsics
    first_c2w = np.linalg.inv(w2c[0])
    w2c = np.stack([pose @ first_c2w for pose in w2c]).astype(np.float32)
    height, width = depth.shape[1:]
    if rgb.shape[1:3] != (height, width):
        rgb = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in rgb])
    jpeg = np.asarray(
        [cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes() for frame in rgb],
        dtype=object,
    )
    # DAVIS does not provide verified capture timestamps here. Seconds below are
    # nominal MotionForesight slots; source_frame_indices remain authoritative.
    slot_seconds = np.arange(len(frame_indices), dtype=np.float64) / 15.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images_jpeg_bytes=jpeg,
        rgb=rgb,
        depth_map=depth,
        extrinsics_w2c=w2c,
        fx_fy_cx_cy=np.asarray(
            [intrinsics[0, 0, 0], intrinsics[0, 1, 1], intrinsics[0, 0, 2], intrinsics[0, 1, 2]],
            dtype=np.float64,
        ),
        frame_indices=frame_indices.astype(np.int32),
        timestamps_s=slot_seconds,
        observed_frame_count=np.asarray(observed_count, dtype=np.int32),
        time_basis=np.asarray("consecutive_davis_source_frame_index"),
        nominal_model_slot_fps=np.asarray(15.0, dtype=np.float64),
        preview_mp4_fps_used=np.asarray(False),
    )
    return {"rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape), "resolution_hw": [height, width]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--masks-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=14)
    ap.add_argument("--t0-frame", type=int, default=20)
    ap.add_argument("--end-frame", type=int, default=35)
    ap.add_argument("--logical-instance-id", type=int, default=1)
    ap.add_argument("--process-res", type=int, default=392)
    args = ap.parse_args()

    indices = np.arange(args.start_frame, args.end_frame + 1, dtype=np.int32)
    if len(indices) != 22 or args.t0_frame != args.start_frame + 6:
        raise ValueError("expected a 22-frame [t0-6, ..., t0+15] window")
    frames = []
    for index in indices:
        bgr = cv2.imread(str(args.frames_dir / f"{index:05d}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.frames_dir / f"{index:05d}.jpg")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("cuda").eval()
    with torch.no_grad():
        observed_prediction = model.inference(frames[:7], process_res=args.process_res)
    observed_path = args.output_dir / "observed7_user.npz"
    observed_shapes = save_user_npz(observed_path, observed_prediction, indices[:7], 7)
    del observed_prediction
    torch.cuda.empty_cache()
    with torch.no_grad():
        full_prediction = model.inference(frames, process_res=args.process_res)
    full_path = args.output_dir / "full22_user.npz"
    full_shapes = save_user_npz(full_path, full_prediction, indices, 7)

    observed = np.load(observed_path, allow_pickle=True)
    height, width = observed["depth_map"].shape[1:]
    mask_path = args.masks_dir / f"{args.t0_frame:05d}.png"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    source_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    if source_mask.ndim != 2:
        raise ValueError(f"expected indexed DAVIS palette PNG, got {source_mask.shape}")
    target_source = source_mask == args.logical_instance_id
    target = cv2.resize(target_source.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    target = cv2.erode(target, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    last_three_depth = np.asarray(observed["depth_map"][-3:])
    depth_valid = np.isfinite(last_three_depth).all(0) & (last_three_depth > 0).all(0)
    candidate = target & depth_valid
    ys, xs = np.nonzero(target)
    if len(xs) == 0:
        raise ValueError("empty eroded target mask")
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    bbox_diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    minimum = max(4.0, 0.02 * bbox_diag)
    query_t0 = deterministic_fps(candidate, 48, minimum)
    source_h, source_w = target_source.shape
    query_source = query_t0 * np.asarray([source_w / width, source_h / height], dtype=np.float32)
    manifest = {
        "video_id": args.sequence,
        "source_dataset": "DAVIS-2017-trainval-480p",
        "logical_instance_id": args.logical_instance_id,
        "png_palette_index": args.logical_instance_id,
        "opencv_bgr_rendering_for_palette_index_1": [0, 0, 128],
        "source_frame_indices": indices.tolist(),
        "observed_source_frame_indices": indices[:7].tolist(),
        "offline_future_source_frame_indices": indices[7:].tolist(),
        "t0_source_frame_index": args.t0_frame,
        "t0_dense_frame_index": 6,
        "time_basis": "consecutive DAVIS source frame indices",
        "preview_mp4_fps_used": False,
        "nominal_motionforesight_slot_fps": 15.0,
        "mask_shape_hw": [height, width],
        "source_mask_shape_hw": [source_h, source_w],
        "bbox_xyxy": bbox,
        "bbox_diagonal_px": bbox_diag,
        "erosion_pixels_at_processed_resolution": 2,
        "candidate_pixel_count": int(candidate.sum()),
        "min_query_distance_px": minimum,
        "query_sets": {"48": query_t0.tolist()},
        "query_uv_t0_source_image": query_source.tolist(),
    }
    (args.output_dir / "query_t0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload = {
        "status": "PASS",
        "sequence": args.sequence,
        "observed": observed_shapes,
        "offline_full": full_shapes,
        "query_count": 48,
        "candidate_pixel_count": int(candidate.sum()),
        "time_basis": manifest["time_basis"],
        "preview_mp4_fps_used": False,
    }
    (args.output_dir / "input_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
