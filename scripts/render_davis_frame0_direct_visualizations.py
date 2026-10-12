#!/usr/bin/env python3
"""Render source-frame-14 direct queries and existing corrected trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from render_davis_track_visualizations import (
    draw_mask_boundary, draw_tracks, render_comparison, render_video, scale_uv,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--encode-fps", type=float, default=8.0)
    args = ap.parse_args()
    v = args.video_id
    d = args.probe_root / v
    out = args.output_root / v
    out.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.rgb_root / v
    mask_dir = args.mask_root / v
    with np.load(d / "visualization_projection.npz") as p:
        indices = [int(x) for x in p["source_frame_indices"]]
        obs_indices = [int(x) for x in p["observed_source_frame_indices"]]
        processed_hw = tuple(int(x) for x in p["processed_resolution_hw"])
        first_frame = cv2.imread(str(rgb_dir / "00014.jpg"), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise FileNotFoundError(rgb_dir / "00014.jpg")
        raw_hw = first_frame.shape[:2]
        tu = scale_uv(p["teacher_uv_processed"], processed_hw, raw_hw)
        ou = scale_uv(p["observed_uv_processed"], processed_hw, raw_hw)
        tv = np.asarray(p["teacher_valid"], dtype=bool)
        ov = np.asarray(p["observed_valid"], dtype=bool)
        q = np.asarray(p["query_uv_frame0_source"], dtype=np.float64)
    if indices != list(range(14, 36)) or obs_indices != list(range(14, 21)) or q.shape != (48, 2):
        raise ValueError("incorrect source frame or query protocol")
    quality = json.loads((d / "quality_audit.json").read_text(encoding="utf-8"))
    teacher = out / "teacher_full22_trackcraft_query_overlay.mp4"
    observed = out / "observed7_trackcraft_query_overlay.mp4"
    frame_paths = [rgb_dir / f"{i:05d}.jpg" for i in indices]
    teacher_meta = render_video(frame_paths, mask_dir, tu, tv, indices, teacher, (40, 60, 255), "FRAME14 QUERY / FULL22 TEACHER", args.encode_fps)
    observed_meta = render_video(frame_paths[:7], mask_dir, ou, ov, obs_indices, observed, (255, 180, 0), "FRAME14 QUERY / OBSERVED7", args.encode_fps)
    frame0 = out / "frame0_source14_48_queries.png"
    image = first_frame.copy()
    draw_mask_boundary(image, mask_dir / "00014.png")
    count = draw_tracks(image, q, np.isfinite(q).all(-1), None, None, (30, 220, 30), labels=True)
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(image, "SOURCE FRAME 14 / MODEL FRAME 0 / DIRECT MASK QUERIES 0-47", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1, cv2.LINE_AA)
    if count != 48 or not cv2.imwrite(str(frame0), image):
        raise RuntimeError(f"frame0 image failed, rendered={count}")
    t0 = out / "t0_source20_teacher_observed_query_comparison.png"
    comparison = render_comparison(rgb_dir / "00020.jpg", mask_dir / "00020.png", tu[6], tv[6], ou[6], ov[6], t0)
    products = [teacher, observed, frame0, t0]
    visible_all = all(n == 48 for n in teacher_meta["rendered_valid_points_per_frame"])
    payload = {
        "video_id": v,
        "visualization_status": "COMPLETE" if visible_all else "COMPLETE_PARTIAL_FAIL_OUT_OF_FRAME",
        "quality_gate_status": quality["status"],
        "query_reference_source_frame_index": 14,
        "output_camera_source_frame_index": 20,
        "source_frame_indices": indices,
        "observed_source_frame_indices": obs_indices,
        "preview_encode_fps_playback_only": args.encode_fps,
        "capture_fps_known": False,
        "trajectory_source": {"teacher": "new full22_query_tracks.npz frame14 direct query", "observed": "new observed7_query_tracks.npz frame14 direct query"},
        "teacher_video": teacher_meta,
        "observed_video": observed_meta,
        "frame0_query_count_rendered": count,
        "t0_comparison": comparison,
        "product_sha256": {f.name: digest(f) for f in products},
    }
    (out / "visualization_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video_id": v, "visualization_status": payload["visualization_status"], "quality_gate_status": payload["quality_gate_status"], "frame0_query_count": count}))


if __name__ == "__main__":
    main()
