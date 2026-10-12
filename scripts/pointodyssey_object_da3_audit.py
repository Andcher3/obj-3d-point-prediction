#!/usr/bin/env python3
"""Audit exact target-object GT registration to RGB-only DA3 observed t0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pointodyssey_label_registration_audit import residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--user-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    query = json.loads((args.object_dir / "query_manifest.json").read_text(encoding="utf-8"))
    bundle = np.load(args.object_dir / "pointodyssey_gt_bundle.npz")
    user = np.load(args.user_npz, allow_pickle=True)
    uv = np.asarray(query["query_sets"]["48"], dtype=np.float64)
    gt = np.asarray(bundle["points_3d_history"][-1], dtype=np.float64)
    depth = np.asarray(user["depth_map"][-1], dtype=np.float64)
    height, width = map(int, query["mask_shape_hw"])
    depth_h, depth_w = depth.shape
    xy = uv * np.asarray([depth_w / width, depth_h / height])
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth_w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth_h - 1)
    z = depth[y, x]
    fx, fy, cx, cy = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
    da3_t0 = np.stack(((xy[:, 0] - cx) / fx * z, (xy[:, 1] - cy) / fy * z, z), axis=-1)
    result = {
        "video_id": query["video_id"],
        "target_mask_ids": query["target_mask_ids"],
        "query_count": len(uv),
        "future_gt_valid_min": int(bundle["validity"][7:].sum(axis=1).min()),
        "future_visible_min": int(bundle["visibility"][7:].sum(axis=1).min()),
        "da3_direct_t0": residuals(gt, da3_t0),
        "gt_used_as_model_input": False,
    }
    result["registration_gate_pass"] = bool(
        result["da3_direct_t0"]["held_out_residual_p90_bbox_fraction"] <= 0.1
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
