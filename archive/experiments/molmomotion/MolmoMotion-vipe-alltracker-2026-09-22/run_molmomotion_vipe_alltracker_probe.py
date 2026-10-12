#!/usr/bin/env python3
"""Run MolmoMotion with 3D histories produced by ViPE and AllTracker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def transform(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((*xyz.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate([xyz.astype(np.float32), ones], axis=-1)
    return np.einsum("ij,...j->...i", matrix.astype(np.float32), homogeneous)[..., :3]


def read_video(path: Path) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def motion_metrics(future: torch.Tensor, anchor: torch.Tensor) -> dict:
    future = future.detach().float().cpu()
    anchor = anchor.detach().float().cpu()
    full = torch.cat([anchor[:, None], future], dim=1)
    steps = torch.linalg.vector_norm(full[:, 1:] - full[:, :-1], dim=-1)
    centroid_delta = future[:, -1].mean(0) - anchor.mean(0)
    return {
        "mean_path_length_m": float(steps.sum(1).mean()),
        "centroid_final_delta_m": centroid_delta.tolist(),
        "finite_fraction": float(torch.isfinite(future).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks-2d", type=Path, required=True)
    parser.add_argument("--tracks-3d", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--future-horizon", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = args.output_dir / "arrays"
    text_dir = args.output_dir / "future_text"
    arrays_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)

    frames = read_video(args.video)
    track_2d = np.load(args.tracks_2d)
    track_3d = np.load(args.tracks_3d)
    poses_c2w = np.load(args.pose)["data"].astype(np.float32)
    intr = np.load(args.intrinsics)["data"].astype(np.float32)
    tracks_uv = track_2d["tracks"].astype(np.float32)
    points_world = track_3d["points_3d"].astype(np.float32).transpose(1, 0, 2)
    visibility = track_3d["visibility"][..., 0].transpose(1, 0)

    t0 = args.history_size - 1
    if len(frames) != len(points_world) or len(frames) != len(tracks_uv):
        raise ValueError(f"frame mismatch: rgb={len(frames)}, 2d={len(tracks_uv)}, 3d={len(points_world)}")
    keep = visibility[: args.history_size].all(0) & np.isfinite(points_world).all(axis=(0, 2))
    if int(keep.sum()) != 8:
        raise RuntimeError(f"expected 8 valid query points, found {int(keep.sum())}")

    world_to_t0 = np.linalg.inv(poses_c2w[t0]).astype(np.float32)
    points_t0_camera = transform(world_to_t0, points_world[:, keep])
    history_xyz = points_t0_camera[: args.history_size]
    pseudo_gt = points_t0_camera[args.history_size :].transpose(1, 0, 2)
    uv_t0 = tracks_uv[t0, keep]
    fx, fy, cx, cy = intr[t0]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    history_frames = frames[: args.history_size]

    sys.path.insert(0, str(args.repo / "src"))
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(str(args.model))
    model = MolmoMotion.from_pretrained(str(args.model))
    model._internal = model._internal.to(torch.bfloat16).cuda()
    if processor.config.history_size != args.history_size:
        raise ValueError(
            f"checkpoint history={processor.config.history_size}, requested={args.history_size}"
        )

    prompts = json.loads(args.spec_json.read_text(encoding="utf-8"))["samples"][0]["prompts"]
    records = []
    futures = {}
    for prompt in prompts:
        inputs_cpu = processor(
            history_frames=history_frames,
            points_2d_at_t0=torch.from_numpy(uv_t0).float(),
            points_3d_history=torch.from_numpy(history_xyz).float(),
            action=prompt["text"],
            future_horizon=args.future_horizon,
        )
        inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs_cpu.items()}
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.predict_trajectory(**inputs)
        elapsed = time.perf_counter() - started
        future = result.future_3d.detach().float().cpu()
        name = prompt["name"]
        futures[name] = future
        row = {
            "prompt": name,
            "prompt_text": prompt["text"],
            "seconds": elapsed,
            "shape": list(future.shape),
            "future_array_sha256": sha_tensor(future),
            "future_text_sha256": hashlib.sha256(result.future_text.encode()).hexdigest(),
            **motion_metrics(future, torch.from_numpy(history_xyz[-1])),
        }
        records.append(row)
        np.savez_compressed(
            arrays_dir / f"{name}__greedy.npz",
            future_3d=future.numpy(),
            pseudo_gt=pseudo_gt,
            points_3d_history=history_xyz,
            points_2d_at_t0=uv_t0,
            intrinsics=k,
            action=np.asarray(prompt["text"]),
        )
        (text_dir / f"{name}__greedy.txt").write_text(result.future_text, encoding="utf-8")
        print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

    pairwise = {}
    names = list(futures)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            delta = torch.linalg.vector_norm(futures[left] - futures[right], dim=-1)
            pairwise[f"{left}__vs__{right}"] = {
                "mean_point_distance_m": float(delta.mean()),
                "max_point_distance_m": float(delta.max()),
            }

    rgb = np.stack([np.asarray(frame) for frame in frames], axis=0)
    np.savez_compressed(args.output_dir / "visualization_input.npz", rgb=rgb)
    payload = {
        "protocol": {
            "history_size": args.history_size,
            "history_frame_indices": list(range(args.history_size)),
            "future_horizon": args.future_horizon,
            "coordinate_frame": "ViPE world points transformed into camera coordinates at history frame 2",
            "vipe_instance_segmentation": False,
            "query_point_count": int(keep.sum()),
        },
        "input_diagnostics": {
            "history_centroid_step_m": np.linalg.norm(
                history_xyz[1:].mean(1) - history_xyz[:-1].mean(1), axis=-1
            ).tolist(),
            "history_mean_point_step_m": np.linalg.norm(
                history_xyz[1:] - history_xyz[:-1], axis=-1
            ).mean(1).tolist(),
            "history_mean_pixel_step": np.linalg.norm(
                tracks_uv[1 : args.history_size, keep] - tracks_uv[: args.history_size - 1, keep], axis=-1
            ).mean(1).tolist(),
            "history_xyz_min": history_xyz.min(axis=(0, 1)).tolist(),
            "history_xyz_max": history_xyz.max(axis=(0, 1)).tolist(),
        },
        "records": records,
        "pairwise": pairwise,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"METRICS={args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
