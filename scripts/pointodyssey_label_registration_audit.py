#!/usr/bin/env python3
"""Held-out t0 registration audit for exact PointOdyssey labels vs RGB-only P0/H."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src = source - src_mean
    dst = target - dst_mean
    u, singular, vt = np.linalg.svd(src.T @ dst)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(src ** 2))
    shift = dst_mean - scale * src_mean @ rotation
    return scale, rotation, shift


def residuals(source: np.ndarray, target: np.ndarray) -> dict:
    scale, rotation, shift = fit_similarity(source, target)
    full = scale * source @ rotation + shift
    bbox = float(np.linalg.norm(np.ptp(full, axis=0)))
    errors = np.linalg.norm(full - target, axis=1) / max(bbox, 1e-9)
    oof_errors = np.full(len(source), np.nan)
    for fold in range(4):
        holdout = np.arange(len(source)) % 4 == fold
        s, r, t = fit_similarity(source[~holdout], target[~holdout])
        oof_errors[holdout] = np.linalg.norm(s * source[holdout] @ r + t - target[holdout], axis=1) / max(bbox, 1e-9)
    return {
        "t0_bbox_diagonal_in_model_units": bbox,
        "fit_scale_gt_to_model": scale,
        "in_sample_residual_median_bbox_fraction": float(np.median(errors)),
        "in_sample_residual_p90_bbox_fraction": float(np.quantile(errors, 0.9)),
        "held_out_residual_median_bbox_fraction": float(np.median(oof_errors)),
        "held_out_residual_p90_bbox_fraction": float(np.quantile(oof_errors, 0.9)),
        "held_out_residual_max_bbox_fraction": float(np.max(oof_errors)),
        "held_out_residual_above_10_percent_count": int(np.sum(oof_errors > 0.1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--user-npz", type=Path)
    parser.add_argument("--source-image-hw", type=int, nargs=2, default=(540, 960))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = np.load(args.cache, allow_pickle=True)
    bundle = np.load(args.ground_truth)
    source = np.asarray(bundle["points_3d_history"][-1], dtype=np.float64)
    target = np.asarray(cache["observed_only_trackcraft_t0"], dtype=np.float64)
    if source.shape != target.shape or source.shape != (48, 3):
        raise ValueError(f"unexpected shapes: {source.shape}, {target.shape}")
    trackcraft_fit = residuals(source, target)
    anchor = np.asarray(cache["query_anchor_match_error_px"], dtype=np.float64)
    result = {
        "source": str(args.ground_truth),
        "cache": str(args.cache),
        "query_count": int(len(source)),
        **trackcraft_fit,
        "anchor_match_p90_dense_px": float(np.quantile(anchor, 0.9)),
        "anchor_match_max_dense_px": float(np.max(anchor)),
        "registration_gate_pass": bool(trackcraft_fit["held_out_residual_p90_bbox_fraction"] <= 0.1 and np.quantile(anchor, 0.9) <= 5.0),
    }
    if args.user_npz is not None:
        user = np.load(args.user_npz, allow_pickle=True)
        depth = np.asarray(user["depth_map"][-1], dtype=np.float64)
        fy_h, fx_w = depth.shape
        uv = np.asarray(cache["query_uv_t0_source"], dtype=np.float64)
        source_h, source_w = args.source_image_hw
        xy = uv * np.asarray([fx_w / source_w, fy_h / source_h])
        x = np.clip(np.rint(xy[:, 0]).astype(int), 0, fx_w - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(int), 0, fy_h - 1)
        z = depth[y, x]
        fx, fy, cx, cy = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
        da3_t0 = np.stack(((xy[:, 0] - cx) / fx * z, (xy[:, 1] - cy) / fy * z, z), axis=-1)
        result["da3_direct_t0"] = residuals(source, da3_t0)
        result["trackcraft_vs_da3_direct_t0_median_units"] = float(np.median(np.linalg.norm(target - da3_t0, axis=1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
