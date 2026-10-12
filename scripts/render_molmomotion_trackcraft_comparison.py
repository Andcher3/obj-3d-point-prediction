#!/usr/bin/env python3
"""Render MolmoMotion future 3D trajectories on the last observed RGB frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (40, 40, 255),
    (40, 180, 255),
    (40, 220, 40),
    (255, 180, 40),
    (255, 60, 60),
    (220, 40, 220),
    (200, 200, 40),
    (255, 255, 255),
]

LABELS = {
    "mouse": "Place bottle beside mouse",
    "black_cup": "Place bottle beside black cup",
    "laptop": "Place bottle beside laptop",
    "keep_holding": "Keep holding bottle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-npz", type=Path, required=True)
    parser.add_argument("--arrays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-index", type=int, default=2)
    parser.add_argument("--fps", type=float, default=6.0)
    return parser.parse_args()


def project(points_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points_3d[..., 2:3], 1e-6, None)
    uv = points_3d[..., :2] / z
    uv[..., 0] = uv[..., 0] * intrinsics[0, 0] + intrinsics[0, 2]
    uv[..., 1] = uv[..., 1] * intrinsics[1, 1] + intrinsics[1, 2]
    return uv


def draw_panel(base_bgr: np.ndarray, data: dict[str, np.ndarray], name: str, step: int) -> np.ndarray:
    frame = base_bgr.copy()
    future_3d = data["future_3d"]
    intrinsics = data["intrinsics"]
    future_2d = project(future_3d, intrinsics)
    history_2d = data["points_2d_at_t0"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.putText(frame, LABELS.get(name, name), (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"future step {step + 1:02d}/{future_3d.shape[1]:02d}",
                (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    for point_i, color in enumerate(COLORS[: future_3d.shape[0]]):
        start = tuple(np.rint(history_2d[point_i]).astype(int))
        cv2.circle(frame, start, 6, color, 2, cv2.LINE_AA)
        path = np.concatenate([history_2d[point_i][None], future_2d[point_i, : step + 1]], axis=0)
        path_i = np.rint(path).astype(np.int32)
        if len(path_i) > 1:
            cv2.polylines(frame, [path_i], False, color, 2, cv2.LINE_AA)
        end = tuple(path_i[-1])
        cv2.circle(frame, end, 4, color, -1, cv2.LINE_AA)

    cv2.putText(frame, "rings: last observed | dots: predicted", (18, frame.shape[0] - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dense = np.load(args.dense_npz)
    rgb = dense["rgb"][args.history_index]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    data_by_name: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(args.arrays_dir.glob("*__greedy.npz")):
        name = path.stem.removesuffix("__greedy")
        with np.load(path) as loaded:
            data_by_name[name] = {key: loaded[key] for key in loaded.files}

    order = [name for name in ("mouse", "black_cup", "laptop", "keep_holding") if name in data_by_name]
    if len(order) != 4:
        raise RuntimeError(f"Expected four prompt outputs, found: {order}")

    steps = data_by_name[order[0]]["future_3d"].shape[1]
    all_panel_frames: dict[str, list[np.ndarray]] = {name: [] for name in order}
    grid_frames: list[np.ndarray] = []
    for step in range(steps):
        panels = []
        for name in order:
            panel = draw_panel(base_bgr, data_by_name[name], name, step)
            all_panel_frames[name].append(panel)
            panels.append(panel)
        top = np.concatenate(panels[:2], axis=1)
        bottom = np.concatenate(panels[2:], axis=1)
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    for name, frames in all_panel_frames.items():
        write_video(args.output_dir / f"{name}.mp4", frames, args.fps)
        cv2.imwrite(str(args.output_dir / f"{name}_final.png"), frames[-1])
    write_video(args.output_dir / "comparison_2x2.mp4", grid_frames, args.fps)
    cv2.imwrite(str(args.output_dir / "comparison_2x2_final.png"), grid_frames[-1])

    futures = {name: data_by_name[name]["future_3d"] for name in order}
    reference = futures[order[0]]
    report = {
        "prompt_order": order,
        "future_shape": list(reference.shape),
        "max_abs_temporal_change_m": {
            name: float(np.max(np.abs(value[:, 1:] - value[:, :-1]))) for name, value in futures.items()
        },
        "max_abs_difference_from_mouse_m": {
            name: float(np.max(np.abs(value - reference))) for name, value in futures.items()
        },
    }
    (args.output_dir / "visualization_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
