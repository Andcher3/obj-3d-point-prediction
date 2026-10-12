#!/usr/bin/env python3
"""Convert TrackCraft3R dense tracks into MolmoMotion inputs and probe text control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


TEXT_VARIANTS = {
    "compact_right": "Pick up the blue-handled scissors and move them to the right.",
    "free_right": (
        "The person grasps the blue-handled scissors from the table and carries them "
        "steadily toward the right side of the sewing machine."
    ),
    "multi_step": (
        "First pick up the blue-handled scissors, move them to the right side of the "
        "sewing machine, and then bring them slightly toward the camera."
    ),
    "contradictory_left": (
        "Leave the blue-handled scissors on the table and slide them to the left."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--camera-npz", required=True)
    p.add_argument("--dense-npz", required=True)
    p.add_argument("--query-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--future-horizon", type=int, default=30)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--variants", nargs="+", default=list(TEXT_VARIANTS))
    return p.parse_args()


def import_quickstart(repo: Path):
    path = repo / "examples" / "01_quickstart.py"
    spec = importlib.util.spec_from_file_location("molmo_motion_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def transform_points(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1), np.float32)], axis=-1)
    return np.einsum("ij,...j->...i", matrix.astype(np.float32), homogeneous)[..., :3]


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = k[0, 0] * xyz[..., 0] / z + k[0, 2]
    v = k[1, 1] * xyz[..., 1] / z + k[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def trajectory_metrics(future: torch.Tensor, t0_xyz: torch.Tensor) -> dict:
    future = future.detach().float().cpu()
    t0_xyz = t0_xyz.detach().float().cpu()
    full = torch.cat([t0_xyz[:, None, :], future], dim=1)
    steps = torch.linalg.vector_norm(full[:, 1:] - full[:, :-1], dim=-1)
    centroid_delta = future[:, -1].mean(0) - t0_xyz.mean(0)
    generated_steps = torch.linalg.vector_norm(future[:, 1:] - future[:, :-1], dim=-1)
    return {
        "finite_fraction": float(torch.isfinite(future).float().mean()),
        "mean_path_length_m": float(steps.sum(dim=1).mean()),
        "centroid_final_delta_m": centroid_delta.tolist(),
        "centroid_final_dx_m": float(centroid_delta[0]),
        "generated_temporal_path_length_m": float(generated_steps.sum(dim=1).mean()),
    }


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = torch.linalg.vector_norm(a.float() - b.float(), dim=-1)
    va = a[:, -1].mean(0) - a[:, 0].mean(0)
    vb = b[:, -1].mean(0) - b[:, 0].mean(0)
    denom = torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)
    cosine = float(torch.dot(va, vb) / denom) if float(denom) > 1e-12 else math.nan
    return {"ade_m": float(delta.mean()), "fde_m": float(delta[:, -1].mean()), "direction_cosine": cosine}


def save_input_overlay(path: Path, image: Image.Image, uv: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.imshow(image)
    ax.scatter(uv[:, 0], uv[:, 1], c=np.arange(len(uv)), cmap="tab10", s=36, edgecolors="white")
    for i, (u, v) in enumerate(uv):
        ax.text(float(u) + 4, float(v) - 4, str(i), color="white", fontsize=7)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_prediction_overlay(path: Path, image: Image.Image, k: np.ndarray, preds: dict[str, torch.Tensor]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.imshow(image)
    for color, (name, future) in zip(plt.cm.tab10(np.linspace(0, 1, len(preds))), preds.items()):
        pixels = project(future.detach().float().cpu().numpy(), k)
        centroid = pixels.mean(axis=0)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, linewidth=2, label=name)
        ax.scatter(centroid[-1, 0], centroid[-1, 1], color=color, s=22)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    (output / "future_text").mkdir(exist_ok=True)
    sys.path.insert(0, str(repo / "src"))
    quickstart = import_quickstart(repo)
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    camera = np.load(args.camera_npz, allow_pickle=True)
    dense = np.load(args.dense_npz)
    track_map = np.asarray(dense["track_map"], np.float32)
    rgb = np.asarray(dense["rgb"], np.uint8)
    extrinsics = np.asarray(camera["extrinsics_w2c"], np.float32)
    fx, fy, cx, cy = np.asarray(camera["fx_fy_cx_cy"], np.float32)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    query_spec = json.loads(Path(args.query_json).read_text(encoding="utf-8"))
    query_uv0 = np.asarray(query_spec["frame0_query_pixels"], np.int64)

    if track_map.shape[:3] != rgb.shape[:3]:
        raise ValueError(f"track/rgb mismatch: {track_map.shape} vs {rgb.shape}")
    if len(track_map) < args.history_size:
        raise ValueError("Tracker output is shorter than MolmoMotion history")
    t0 = len(track_map) - 1
    history_indices = np.arange(t0 - args.history_size + 1, t0 + 1)
    u0 = query_uv0[:, 0]
    v0 = query_uv0[:, 1]
    if np.any(u0 < 0) or np.any(u0 >= track_map.shape[2]) or np.any(v0 < 0) or np.any(v0 >= track_map.shape[1]):
        raise ValueError("frame-0 query lies outside TrackCraft3R output")

    xyz_frame0 = np.stack([track_map[t, v0, u0] for t in history_indices], axis=0)
    xyz_t0 = transform_points(extrinsics[t0], xyz_frame0)
    uv_t0 = project(xyz_t0[-1], k)
    finite = np.isfinite(xyz_t0).all(axis=(0, 2)) & np.isfinite(uv_t0).all(axis=1)
    inside = (uv_t0[:, 0] >= 0) & (uv_t0[:, 0] < rgb.shape[2]) & (uv_t0[:, 1] >= 0) & (uv_t0[:, 1] < rgb.shape[1])
    positive_depth = (xyz_t0[..., 2] > 1e-4).all(axis=0)
    keep = finite & inside & positive_depth
    if keep.sum() < 4:
        raise RuntimeError(f"Only {int(keep.sum())} valid target tracks remain; need at least 4")
    query_uv0 = query_uv0[keep]
    uv_t0 = uv_t0[keep]
    xyz_t0 = xyz_t0[:, keep]

    history_frames = [Image.fromarray(rgb[t]) for t in history_indices]
    points_2d_at_t0 = torch.from_numpy(uv_t0).float()
    points_3d_history = torch.from_numpy(xyz_t0).float()
    intrinsics = torch.from_numpy(k).float()
    save_input_overlay(output / "input_query_points.png", history_frames[-1], uv_t0)

    processor = MolmoMotionProcessor.from_pretrained(args.model)
    if processor.config.history_size != args.history_size:
        raise ValueError(f"Checkpoint history={processor.config.history_size}, requested={args.history_size}")
    model = MolmoMotion.from_pretrained(args.model)
    model._internal = model._internal.to(torch.bfloat16).cuda()

    records = []
    predictions: dict[str, torch.Tensor] = {}
    for name in args.variants:
        action = TEXT_VARIANTS[name]
        inputs = processor(
            history_frames=history_frames,
            points_2d_at_t0=points_2d_at_t0,
            points_3d_history=points_3d_history,
            action=action,
            future_horizon=args.future_horizon,
        )
        input_ids = inputs["input_ids"]
        input_ids_sha256 = tensor_sha256(input_ids)
        input_token_count = int(input_ids.numel())
        inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.predict_trajectory(**inputs)
        future = result.future_3d.detach().float().cpu()
        future_text = result.future_text
        elapsed = time.perf_counter() - started
        expected = (len(uv_t0), args.future_horizon, 3)
        if tuple(future.shape) != expected:
            raise RuntimeError(f"Unexpected output for {name}: {tuple(future.shape)} != {expected}")
        predictions[name] = future
        row = {
            "variant": name,
            "action": action,
            "shape": list(future.shape),
            "seconds": elapsed,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "input_token_count": input_token_count,
            "input_ids_sha256": input_ids_sha256,
            "future_text_chars": len(future_text),
            "future_text_sha256": hashlib.sha256(future_text.encode("utf-8")).hexdigest(),
            **trajectory_metrics(future, points_3d_history[-1]),
        }
        records.append(row)
        np.savez_compressed(
            output / "arrays" / f"{name}.npz",
            future_3d=future.numpy(),
            points_3d_history=xyz_t0,
            points_2d_at_t0=uv_t0,
            frame0_query_pixels=query_uv0,
            intrinsics=k,
            history_indices=history_indices,
            action=np.asarray(action),
        )
        (output / "future_text" / f"{name}.txt").write_text(future_text, encoding="utf-8")
        print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

    pairs = {}
    names = list(predictions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}__vs__{b}"] = compare(predictions[a], predictions[b])
    save_prediction_overlay(output / "prediction_centroids.png", history_frames[-1], k, predictions)
    payload = {
        "trackcraft_dense": str(Path(args.dense_npz).resolve()),
        "camera_npz": str(Path(args.camera_npz).resolve()),
        "query_spec": query_spec,
        "history_indices": history_indices.tolist(),
        "valid_query_count": int(len(uv_t0)),
        "records": records,
        "pairwise": pairs,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VALID_QUERY_COUNT={len(uv_t0)}")
    print(f"METRICS={output / 'metrics.json'}")


if __name__ == "__main__":
    main()
