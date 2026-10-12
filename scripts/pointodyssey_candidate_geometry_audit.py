#!/usr/bin/env python3
"""Measure RGB-only DA3 t0 geometry quality for all viable t0 mask segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pointodyssey_interface_preflight import farthest_points
from pointodyssey_label_registration_audit import residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--user-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t0-frame", type=int, default=12)
    args = parser.parse_args()

    anno = np.load(args.sequence / "anno.npz")
    user = np.load(args.user_npz, allow_pickle=True)
    mask = np.asarray(Image.open(args.sequence / "masks" / f"mask_{args.t0_frame:05d}.png"))
    all_uv = anno["trajs_2d"]
    uv = np.asarray(all_uv[args.t0_frame], dtype=np.float64)
    uv_first = np.asarray(all_uv[0], dtype=np.float64)
    world = np.asarray(anno["trajs_3d"][args.t0_frame], dtype=np.float64)
    camera = np.asarray(anno["extrinsics"][args.t0_frame], dtype=np.float64)
    gt = world @ camera[:3, :3].T + camera[:3, 3]
    valid = np.asarray(anno["valids"][args.t0_frame], dtype=bool)
    visible = np.asarray(anno["visibs"][args.t0_frame], dtype=bool)
    height, width = mask.shape
    finite = np.isfinite(uv).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    x = np.rint(np.where(finite, uv[:, 0], -1)).astype(int)
    y = np.rint(np.where(finite, uv[:, 1], -1)).astype(int)
    keep = finite & valid & visible & (x >= 1) & (x < width - 1) & (y >= 1) & (y < height - 1)
    ids = np.flatnonzero(keep)
    segment = mask[y[ids], x[ids]]
    inner = (
        (mask[y[ids] - 1, x[ids]] == segment)
        & (mask[y[ids] + 1, x[ids]] == segment)
        & (mask[y[ids], x[ids] - 1] == segment)
        & (mask[y[ids], x[ids] + 1] == segment)
    )
    ids = ids[inner]
    segment = segment[inner]
    depth = np.asarray(user["depth_map"][-1], dtype=np.float64)
    depth_h, depth_w = depth.shape
    intr = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
    rows = []
    for label, pixels in zip(*np.unique(mask, return_counts=True)):
        label = int(label)
        area = float(pixels / mask.size)
        eligible = ids[segment == label]
        if label == 0 or not (0.002 <= area <= 0.35) or len(eligible) < 48:
            continue
        point_ids = eligible[farthest_points(uv[eligible], 48)]
        xy = uv[point_ids] * np.asarray([depth_w / width, depth_h / height])
        ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth_w - 1)
        iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth_h - 1)
        z = depth[iy, ix]
        fx, fy, cx, cy = intr
        xyz = np.stack(((xy[:, 0] - cx) / fx * z, (xy[:, 1] - cy) / fy * z, z), axis=-1)
        result = residuals(gt[point_ids], xyz)
        rows.append({
            "mask_id": label,
            "area_fraction_t0": area,
            "t0_candidate_track_count": int(len(eligible)),
            "observed_median_displacement_px": float(np.median(
                np.linalg.norm(uv[point_ids] - uv_first[point_ids], axis=-1)
            )),
            "held_out_p90_bbox_fraction": result["held_out_residual_p90_bbox_fraction"],
            "held_out_above_10_percent_count": result["held_out_residual_above_10_percent_count"],
        })
    rows.sort(key=lambda row: (-row["t0_candidate_track_count"], row["mask_id"]))
    output = {
        "sequence": args.sequence.name,
        "future_labels_used_for_candidate_selection": False,
        "candidate_count": len(rows),
        "candidate_count_registration_p90_le_10_percent": sum(row["held_out_p90_bbox_fraction"] <= 0.1 for row in rows),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "candidates"}, indent=2))
    print("top by t0-only support:", rows[:5])


if __name__ == "__main__":
    main()
