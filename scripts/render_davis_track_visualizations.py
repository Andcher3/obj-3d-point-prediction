#!/usr/bin/env python3
"""Render existing DAVIS TrackCraft3R query trajectories over RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def stable_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(count):
        hue = (i * 0.61803398875) % 1.0
        h = int(hue * 179)
        hsv = np.uint8([[[h, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(x) for x in bgr))
    return colors


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8) == 1


def draw_mask_boundary(image: np.ndarray, mask_path: Path) -> None:
    if not mask_path.exists():
        return
    mask = read_mask(mask_path).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 220, 255), 1, lineType=cv2.LINE_AA)


def put_banner(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    height = 34
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    cv2.putText(image, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)


def scale_uv(uv: np.ndarray, processed_hw: tuple[int, int], raw_hw: tuple[int, int]) -> np.ndarray:
    ph, pw = processed_hw
    rh, rw = raw_hw
    out = np.asarray(uv, dtype=np.float64).copy()
    out[..., 0] *= rw / pw
    out[..., 1] *= rh / ph
    return out


def valid_point(point: np.ndarray, width: int, height: int) -> bool:
    return bool(np.isfinite(point).all() and 0 <= point[0] < width and 0 <= point[1] < height)


def draw_tracks(
    image: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    previous: np.ndarray | None,
    previous_valid: np.ndarray | None,
    color: tuple[int, int, int],
    trails: int = 4,
    labels: bool = False,
) -> int:
    height, width = image.shape[:2]
    count = 0
    if previous is not None and previous_valid is not None:
        for q in range(points.shape[0]):
            if not valid[q] or not previous_valid[q]:
                continue
            if not valid_point(points[q], width, height) or not valid_point(previous[q], width, height):
                continue
            cv2.line(image, tuple(np.rint(previous[q]).astype(int)), tuple(np.rint(points[q]).astype(int)), color, 1, cv2.LINE_AA)
    for q, point in enumerate(points):
        if not valid[q] or not valid_point(point, width, height):
            continue
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(image, xy, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, xy, 2, color, -1, cv2.LINE_AA)
        if labels:
            cv2.putText(image, str(q), (xy[0] + 4, xy[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        count += 1
    return count


def render_video(
    frame_paths: list[Path],
    mask_dir: Path,
    uv: np.ndarray,
    valid: np.ndarray,
    frame_indices: list[int],
    output: Path,
    banner_color: tuple[int, int, int],
    label: str,
    encode_fps: float,
) -> dict:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), encode_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")
    counts = []
    previous = None
    previous_valid = None
    for i, (frame_path, source_index) in enumerate(zip(frame_paths, frame_indices)):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {frame_path}")
        draw_mask_boundary(image, mask_dir / f"{source_index:05d}.png")
        current = uv[i]
        current_valid = valid[i]
        count = draw_tracks(image, current, current_valid, previous, previous_valid, banner_color, labels=(source_index == 20))
        put_banner(image, f"{label} | source frame {source_index} | t0=20 | 48 queries | encode {encode_fps:g} fps playback", banner_color)
        writer.write(image)
        counts.append(count)
        previous = current
        previous_valid = current_valid
    writer.release()
    return {"frame_count": len(frame_paths), "rendered_valid_points_per_frame": counts, "resolution_hw": [height, width], "encode_fps": encode_fps}


def render_comparison(frame_path: Path, mask_path: Path, teacher: np.ndarray, teacher_valid: np.ndarray, observed: np.ndarray, observed_valid: np.ndarray, output: Path) -> dict:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {frame_path}")
    height, width = image.shape[:2]
    draw_mask_boundary(image, mask_path)
    teacher_count = 0
    observed_count = 0
    for q in range(48):
        tv = teacher_valid[q] and valid_point(teacher[q], width, height)
        ov = observed_valid[q] and valid_point(observed[q], width, height)
        if tv and ov:
            a = tuple(np.rint(observed[q]).astype(int))
            b = tuple(np.rint(teacher[q]).astype(int))
            cv2.line(image, a, b, (0, 215, 255), 1, cv2.LINE_AA)
        if ov:
            xy = tuple(np.rint(observed[q]).astype(int))
            cv2.circle(image, xy, 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(image, xy, 3, (255, 180, 0), -1, cv2.LINE_AA)
            cv2.putText(image, f"O{q}", (xy[0] + 4, xy[1] + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 180, 0), 1, cv2.LINE_AA)
            observed_count += 1
        if tv:
            xy = tuple(np.rint(teacher[q]).astype(int))
            cv2.circle(image, xy, 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(image, xy, 3, (40, 60, 255), -1, cv2.LINE_AA)
            cv2.putText(image, f"T{q}", (xy[0] + 4, xy[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 60, 255), 1, cv2.LINE_AA)
            teacher_count += 1
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, "t0 source frame 20 | red T=official interleaved teacher | blue O=observed7 | yellow=line", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "48 stable query IDs; projection from existing TrackCraft3R 3D tracks", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), image)
    return {"resolution_hw": [height, width], "teacher_rendered_count": teacher_count, "observed_rendered_count": observed_count}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--encode-fps", type=float, default=8.0)
    args = ap.parse_args()

    rgb_dir = args.rgb_root / args.sequence
    mask_dir = args.mask_root / args.sequence
    probe_dir = args.probe_root / args.sequence
    out = args.output_root / args.sequence
    out.mkdir(parents=True, exist_ok=True)

    projection = np.load(probe_dir / "visualization_projection.npz", allow_pickle=True)
    quality = json.loads((probe_dir / "quality_audit.json").read_text(encoding="utf-8"))
    source_indices = [int(x) for x in projection["source_frame_indices"]]
    observed_indices = [int(x) for x in projection["observed_source_frame_indices"]]
    processed_hw = tuple(int(x) for x in projection["processed_resolution_hw"])
    frame_paths = [rgb_dir / f"{i:05d}.jpg" for i in source_indices]
    observed_frame_paths = [rgb_dir / f"{i:05d}.jpg" for i in observed_indices]
    for path in frame_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    teacher_uv = scale_uv(projection["teacher_uv_processed"], processed_hw, cv2.imread(str(frame_paths[0])).shape[:2])
    observed_uv = scale_uv(projection["observed_uv_processed"], processed_hw, cv2.imread(str(frame_paths[0])).shape[:2])
    teacher_valid = np.asarray(projection["teacher_valid"], dtype=bool)
    observed_valid = np.asarray(projection["observed_valid"], dtype=bool)
    teacher_video = out / "teacher_official_interleaved22_pj0_overlay.mp4"
    observed_video = out / "observed7_pj0_query_overlay.mp4"
    teacher_meta = render_video(frame_paths, mask_dir, teacher_uv, teacher_valid, source_indices, teacher_video, (40, 60, 255), "OFFICIAL INTERLEAVED TEACHER", args.encode_fps)
    observed_meta = render_video(observed_frame_paths, mask_dir, observed_uv, observed_valid, observed_indices, observed_video, (255, 180, 0), "OBSERVED7 DIAGNOSTIC", args.encode_fps)
    t0_frame = rgb_dir / "00020.jpg"
    comparison = out / "t0_teacher_observed_query_comparison.png"
    comparison_meta = render_comparison(t0_frame, mask_dir / "00020.png", teacher_uv[6], teacher_valid[6], observed_uv[6], observed_valid[6], comparison)

    files = [teacher_video, observed_video, comparison]
    complete_projection = all(count == 48 for count in teacher_meta["rendered_valid_points_per_frame"])
    render_status = "COMPLETE" if complete_projection else "COMPLETE_PARTIAL_FAIL"
    manifest = {
        "status": render_status if all(path.exists() and path.stat().st_size > 0 for path in files) else "FAIL",
        "quality_gate_status": quality["status"],
        "video_id": args.sequence,
        "trajectory_source": {
            "teacher": "full22_query_tracks.npz; two official 12-frame interleaved runs with shared frame0 and per-run Pj0 anchoring",
            "observed": "observed7_query_tracks.npz; nonofficial 7-frame query-only diagnostic with Pj0 anchoring",
            "no_model_rerun": True,
        },
        "time": {
            "source_frame_indices": source_indices,
            "observed_source_frame_indices": observed_indices,
            "t0_source_frame_index": 20,
            "t0_dense_frame_index": 6,
            "time_basis": "consecutive DAVIS source frame index",
            "preview_encode_fps_playback_only": args.encode_fps,
            "source_capture_fps_used": None,
        },
        "query": {"count": 48, "stable_ids": "0..47", "mapping_file": "query_frame0_manifest.json"},
        "source_resolution": teacher_meta["resolution_hw"],
        "teacher_video": {"file": teacher_video.name, **teacher_meta},
        "observed_video": {"file": observed_video.name, **observed_meta},
        "t0_comparison": {"file": comparison.name, **comparison_meta},
        "quality_audit_file": "quality_audit.json",
        "trajectory_projection_status": "COMPLETE" if complete_projection else "PARTIAL_FAIL_OUT_OF_FRAME_PROJECTIONS",
    }
    for path in files:
        manifest.setdefault("sha256", {})[path.name] = sha256(path)
    (out / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
