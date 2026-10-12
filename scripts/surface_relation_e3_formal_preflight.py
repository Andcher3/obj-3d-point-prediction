#!/usr/bin/env python3
"""Run formal observed-only MotionForesight P0/H interface checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def sample_map(values: torch.Tensor, uv: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Sample B,C,T,H,W values at model-pixel anchors and return T,N,C."""
    grid = torch.as_tensor(uv, dtype=torch.float32, device=values.device).clone()
    grid[:, 0] = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
    grid = grid.view(1, 1, -1, 2)
    per_time = values[0].permute(1, 0, 2, 3)
    return F.grid_sample(
        per_time,
        grid.expand(per_time.shape[0], -1, -1, -1),
        mode="bilinear",
        align_corners=True,
    )[:, :, 0].permute(0, 2, 1)


def match_t0_queries(
    track_map: np.ndarray,
    last_w2c: np.ndarray,
    intrinsics: np.ndarray,
    query_uv_t0: np.ndarray,
    query_hw: tuple[int, int],
    intrinsic_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map t0 queries to unique frame-0 pixels using observed-only tracks."""
    height, width = track_map.shape[1:3]
    source_h, source_w = query_hw
    intrinsic_h, intrinsic_w = intrinsic_hw
    k = np.asarray([
        [intrinsics[0] * width / intrinsic_w, 0.0, intrinsics[2] * width / intrinsic_w],
        [0.0, intrinsics[1] * height / intrinsic_h, intrinsics[3] * height / intrinsic_h],
        [0.0, 0.0, 1.0],
    ])
    xyz = transform(last_w2c, track_map[-1])
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-8)
    projected = np.full((*xyz.shape[:2], 2), np.nan, dtype=np.float64)
    projected[..., 0][valid] = k[0, 0] * xyz[..., 0][valid] / xyz[..., 2][valid] + k[0, 2]
    projected[..., 1][valid] = k[1, 1] * xyz[..., 1][valid] / xyz[..., 2][valid] + k[1, 2]
    query_model = query_uv_t0 * np.asarray([width / source_w, height / source_h])
    valid_y, valid_x = np.nonzero(valid)
    tree = cKDTree(projected[valid])
    _, neighbours = tree.query(query_model, k=min(128, len(valid_x)))
    candidates = np.unique(np.asarray(neighbours).reshape(-1))
    costs = cdist(query_model, projected[valid][candidates])
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != len(query_model) or not np.array_equal(np.sort(rows), np.arange(len(query_model))):
        raise RuntimeError("could not assign a global one-to-one frame-0 anchor mapping")
    selected = np.empty(len(query_model), dtype=np.int64)
    selected_distances = np.empty(len(query_model), dtype=np.float32)
    selected[rows] = candidates[columns]
    selected_distances[rows] = costs[rows, columns]
    anchors = np.stack([valid_x[selected], valid_y[selected]], axis=-1).astype(np.float32)
    observed_t0 = xyz[anchors[:, 1].astype(np.int64), anchors[:, 0].astype(np.int64)].astype(np.float32)
    return anchors, selected_distances, observed_t0


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping row-vector source points to target points."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered)
    row_rotation = u @ vt
    if np.linalg.det(row_rotation) < 0:
        u[:, -1] *= -1
        row_rotation = u @ vt
    denominator = float(np.sum(source_centered ** 2))
    scale = float(np.sum(singular_values) / max(denominator, 1e-12))
    translation = target_mean - scale * (source_mean @ row_rotation)
    return scale, row_rotation.astype(np.float32), translation.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--user-npz", type=Path, required=True)
    parser.add_argument("--trackcraft-dense", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--pseudo-gt-bundle", type=Path, required=True)
    parser.add_argument("--cache-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--ground-truth-source", default="trackcraft3r_offline_future")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from scripts.inference_model import build_model_from_ckpt
    from models_pretrained.future_scene_flow.sparse_dataset import (
        _apply_rigid,
        _compute_pj_norm,
        _unproject_world,
    )

    model, state = build_model_from_ckpt(args.checkpoint, args.base_checkpoint)
    cfg = model.config
    if cfg.obs_frames != 7 or cfg.future_frames != 15:
        raise ValueError(f"unexpected checkpoint temporal shape {cfg.obs_frames}+{cfg.future_frames}")
    data = np.load(args.user_npz, allow_pickle=True)
    rgb = np.asarray(data["rgb"])
    depth = np.asarray(data["depth_map"])
    w2c = np.asarray(data["extrinsics_w2c"], dtype=np.float64)
    intr = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
    observed_timestamps = np.asarray(data["timestamps_s"], dtype=np.float64)
    if len(rgb) != cfg.obs_frames or not np.all(np.diff(frame_indices) == 1):
        raise ValueError("formal input must contain seven consecutive observed frames")

    query = json.loads(args.query_json.read_text(encoding="utf-8"))
    query_uv_t0 = np.asarray(query["query_sets"][str(args.query_count)], dtype=np.float32)
    query_hw = tuple(map(int, query["mask_shape_hw"]))
    dense = np.load(args.trackcraft_dense)
    track_map = np.asarray(dense["track_map"], dtype=np.float32)
    if track_map.shape[0] != cfg.obs_frames:
        raise ValueError(f"TrackCraft output must be observed-only, got {track_map.shape}")
    anchors_dense, anchor_errors, observed_t0 = match_t0_queries(
        track_map, w2c[-1], intr, query_uv_t0, query_hw, rgb.shape[1:3]
    )

    pseudo_bundle = np.load(args.pseudo_gt_bundle, allow_pickle=True)
    pseudo_gt_raw = np.asarray(pseudo_bundle["pseudo_gt"][:, : args.query_count], dtype=np.float32)
    pseudo_t0_raw = np.asarray(
        pseudo_bundle["points_3d_history"][-1, : args.query_count], dtype=np.float32
    )
    pseudo_validity_all = np.asarray(pseudo_bundle["validity"][:, : args.query_count], dtype=bool)
    pseudo_validity = pseudo_validity_all[-len(pseudo_gt_raw) :]
    scale, row_rotation, translation = estimate_similarity(pseudo_t0_raw, observed_t0)
    pseudo_t0_aligned = scale * (pseudo_t0_raw @ row_rotation) + translation
    pseudo_gt_aligned = scale * (pseudo_gt_raw @ row_rotation) + translation
    raw_t0_error = np.linalg.norm(pseudo_t0_raw - observed_t0, axis=-1)
    aligned_t0_error = np.linalg.norm(pseudo_t0_aligned - observed_t0, axis=-1)
    column_rotation = row_rotation.T
    rotation_angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(column_rotation) - 1.0) / 2.0, -1.0, 1.0))))

    source_h, source_w = rgb.shape[1:3]
    rgb_model = np.stack([cv2.resize(frame, (cfg.width, cfg.height)) for frame in rgb])
    rgb_model = (rgb_model.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    intr_model = np.asarray([
        intr[0] * cfg.width / source_w,
        intr[1] * cfg.height / source_h,
        intr[2] * cfg.width / source_w,
        intr[3] * cfg.height / source_h,
    ])
    pj_world = np.stack([
        _unproject_world(
            cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_NEAREST).astype(np.float64),
            intr_model,
            pose,
        )
        for frame, pose in zip(depth, w2c)
    ])
    pj = _apply_rigid(w2c[-1], pj_world)
    norm = _compute_pj_norm(pj, 80.0, 2.0, 98.0)
    pj_norm = ((pj - norm.mean) / norm.scale).astype(np.float32)
    p0 = np.transpose(pj_norm[0], (2, 0, 1))
    anchors_model = anchors_dense * np.asarray(
        [cfg.width / track_map.shape[2], cfg.height / track_map.shape[1]], dtype=np.float32
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["last_block"] = output.detach().to("cpu")

    handle = model.dit.blocks[-1].register_forward_hook(hook)
    rgb_t = torch.from_numpy(rgb_model).unsqueeze(0).to(model.device)
    pj_t = torch.from_numpy(np.transpose(pj_norm, (0, 3, 1, 2))).unsqueeze(0).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = model(rgb_t, pj_t)
        xyz_latent, _ = model.split_latent(latent)
        delta = model.decode_xyz(xyz_latent).float()
    handle.remove()
    prediction = model.denormalize(
        model.reconstruct(delta, torch.from_numpy(p0).unsqueeze(0).to(model.device).float()),
        torch.from_numpy(norm.mean).unsqueeze(0).to(model.device).float(),
        torch.tensor([norm.scale], device=model.device),
    )
    sampled_prediction = sample_map(prediction, anchors_model, cfg.height, cfg.width).cpu().numpy()
    sampled_output_h = sample_map(xyz_latent.float(), anchors_model, cfg.height, cfg.width).cpu().numpy()

    hidden = captured["last_block"]
    token_count = hidden.shape[1] // 2
    spatial_tokens = token_count // cfg.total_frames
    token_h = cfg.height // 16
    token_w = spatial_tokens // token_h
    row_hidden = hidden[:, token_count:].reshape(1, cfg.total_frames, token_h, token_w, -1)
    row_hidden = row_hidden.permute(0, 4, 1, 2, 3).float()
    sampled_dit_h = sample_map(row_hidden, anchors_model, cfg.height, cfg.width).cpu().numpy()

    permutation = np.arange(args.query_count)[::-1]
    permuted = sample_map(prediction, anchors_model[permutation], cfg.height, cfg.width).cpu().numpy()
    permutation_error = float(np.max(np.abs(permuted[:, np.argsort(permutation)] - sampled_prediction)))
    future = slice(cfg.obs_frames, cfg.total_frames)
    prediction_timestamps = observed_timestamps[-1] + np.arange(1, cfg.future_frames + 1) / args.fps
    formal_targets = np.asarray(query["future_target_seconds"], dtype=np.float64)
    formal_slot_indices = np.asarray([
        int(np.argmin(np.abs(prediction_timestamps - target))) for target in formal_targets
    ], dtype=np.int32)
    formal_match_errors = np.abs(prediction_timestamps[formal_slot_indices] - formal_targets)
    unique_formal_mapping = len(np.unique(formal_slot_indices)) == len(formal_slot_indices)
    half_step = 0.5 / args.fps
    time_mapping_valid = bool(unique_formal_mapping and np.all(formal_match_errors <= half_step + 1e-9))

    p0_future = sampled_prediction[future]
    h_output = sampled_output_h[future]
    h_dit = sampled_dit_h[future]
    cache_arrays = {
        "p0_future": p0_future,
        "h_output_latent": h_output,
        "h_last_dit": h_dit,
        "p0_formal9": p0_future[formal_slot_indices],
        "h_output_latent_formal9": h_output[formal_slot_indices],
        "h_last_dit_formal9": h_dit[formal_slot_indices],
        "pseudo_gt_raw": pseudo_gt_raw,
        "pseudo_gt_aligned": pseudo_gt_aligned.astype(np.float32),
        "pseudo_gt_validity": pseudo_validity,
        "pseudo_gt_t0_raw": pseudo_t0_raw,
        "pseudo_gt_t0_aligned": pseudo_t0_aligned.astype(np.float32),
        "observed_only_trackcraft_t0": observed_t0,
        "pseudo_gt_alignment_scale": np.asarray(scale, dtype=np.float64),
        "pseudo_gt_alignment_row_rotation": row_rotation,
        "pseudo_gt_alignment_translation": translation,
        "query_uv_t0_source": query_uv_t0,
        "query_uv_frame0_dense": anchors_dense,
        "query_uv_frame0_model": anchors_model,
        "query_anchor_match_error_px": anchor_errors,
        "observed_frame_indices": frame_indices,
        "observed_timestamps_s": observed_timestamps,
        "prediction_timestamps_s": prediction_timestamps,
        "formal_future_target_timestamps_s": formal_targets,
        "formal_future_slot_indices": formal_slot_indices,
        "formal_future_match_errors_s": formal_match_errors,
        "coordinate_frame": np.asarray("last_observed_camera"),
        "prediction_source": np.asarray("motionforesight_observed_only"),
        "pseudo_gt_source": np.asarray(args.ground_truth_source),
        "pseudo_gt_alignment_source": np.asarray("shared_t0_queries_only"),
        "hidden_feature_source_output": np.asarray("decoded_xyz_latent_before_xyz_head"),
        "hidden_feature_source_dit": np.asarray("last_dit_block_row_tokens"),
    }
    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache_npz, **cache_arrays)
    input_hash = hashlib.sha256(rgb_t.cpu().numpy().tobytes() + pj_t.cpu().numpy().tobytes()).hexdigest()
    payload = {
        "status": "PASS" if time_mapping_valid else "FAIL",
        "scope": "formal-video observed-only E3 interface preflight; Relation Head not trained",
        "video_id": query["video_id"],
        "checkpoint_config": state["config"],
        "input": {
            "observed_rgb_shape": list(rgb_t.shape),
            "observed_pointmap_shape": list(pj_t.shape),
            "observed_frame_indices": frame_indices.tolist(),
            "observed_timestamps_s": observed_timestamps.tolist(),
            "observed_tensor_sha256": input_hash,
            "future_frame_tensor_supplied_to_motionforesight": False,
            "trackcraft_identity_mapping_scope": "seven observed frames only",
        },
        "outputs": {
            "p0_future_shape": list(p0_future.shape),
            "h_output_latent_shape": list(h_output.shape),
            "h_last_dit_shape": list(h_dit.shape),
            "finite": bool(np.isfinite(p0_future).all() and np.isfinite(h_output).all() and np.isfinite(h_dit).all()),
            "cache_npz": str(args.cache_npz),
        },
        "query_alignment": {
            "count": args.query_count,
            "anchor": "first observed frame via observed-only TrackCraft3R",
            "match_error_px_median": float(np.median(anchor_errors)),
            "match_error_px_max": float(np.max(anchor_errors)),
            "permutation_sampling_max_abs_error": permutation_error,
        },
        "pseudo_gt_alignment": {
            "method": "least-squares similarity estimated from shared t0 queries only",
            "uses_future_points_for_alignment": False,
            "scale": scale,
            "rotation_angle_deg": rotation_angle_deg,
            "translation_norm": float(np.linalg.norm(translation)),
            "raw_t0_error_median": float(np.median(raw_t0_error)),
            "raw_t0_error_max": float(np.max(raw_t0_error)),
            "aligned_t0_error_median": float(np.median(aligned_t0_error)),
            "aligned_t0_error_max": float(np.max(aligned_t0_error)),
            "pseudo_gt_shape": list(pseudo_gt_aligned.shape),
            "valid_fraction": float(pseudo_validity.mean()),
        },
        "time_mapping": {
            "model_future_slots": cfg.future_frames,
            "prediction_timestamps_s": prediction_timestamps.tolist(),
            "formal_future_slot_indices_zero_based": formal_slot_indices.tolist(),
            "formal_match_errors_s": formal_match_errors.tolist(),
            "tolerance_s": half_step,
            "unique_mapping": unique_formal_mapping,
            "valid": time_mapping_valid,
            "basis": "consecutive stride-1 model slots on an explicitly resampled 15 fps input grid",
            "remaining_limit": "checkpoint training-rate calibration on Ego4D is not established",
        },
        "coordinate_frame": "last_observed_camera",
        "ground_truth_source": args.ground_truth_source,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
