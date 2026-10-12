#!/usr/bin/env python3
"""Export one audited PointOdyssey sequence for RGB-only interface preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pointodyssey_interface_preflight import FRAMES, T0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    matches = [row for row in audit["sequences"] if row["sequence"] == args.sequence.name]
    if len(matches) != 1 or matches[0]["status"] != "PASS":
        raise ValueError("exactly one passing audit row is required")
    row = matches[0]
    ids = np.asarray(row["query_point_indices"], dtype=np.int64)
    anno = np.load(args.sequence / "anno.npz")
    world = np.asarray(anno["trajs_3d"][FRAMES][:, ids], dtype=np.float64)
    camera = np.asarray(anno["extrinsics"][int(FRAMES[T0])], dtype=np.float64)
    gt = world @ camera[:3, :3].T + camera[:3, 3]
    validity = anno["valids"][FRAMES][:, ids].astype(bool)
    visibility = anno["visibs"][FRAMES][:, ids].astype(bool)
    uv_t0 = np.asarray(anno["trajs_2d"][int(FRAMES[T0]), ids], dtype=np.float64)
    if not np.allclose(uv_t0, np.asarray(row["query_uv_t0"]), atol=1e-5):
        raise ValueError("query identity changed since audit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "pointodyssey_gt_bundle.npz",
        points_3d_history=gt[: T0 + 1].astype(np.float32),
        pseudo_gt=gt[T0 + 1 :].astype(np.float32),
        validity=validity,
        visibility=visibility,
        query_point_indices=ids,
        source_frame_indices=FRAMES,
        ground_truth_source=np.asarray("pointodyssey_v1.2_exact_rendered_tracks"),
        coordinate_frame=np.asarray("last_observed_ground_truth_camera"),
    )
    target_seconds = (np.arange(T0 + 1, len(FRAMES)) / 15.0).tolist()
    query = {
        "video_id": args.sequence.name,
        "mask_shape_hw": row["image_shape_hw"],
        "query_sets": {"48": row["query_uv_t0"]},
        "future_target_seconds": target_seconds,
        "target_mask_id": row["selected_mask_id"],
        "target_mask_ids": row["selected_mask_ids"],
        "target_selection_source": row["target_selection_source"],
        "query_source": "t0_only_pointodyssey_mask_and_visible_tracks",
        "ground_truth_source": "pointodyssey_v1.2_exact_rendered_tracks",
    }
    (args.output_dir / "query_manifest.json").write_text(
        json.dumps(query, indent=2) + "\n", encoding="utf-8"
    )
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for observed_index, source_index in enumerate(FRAMES[: T0 + 1]):
        image = np.asarray(Image.open(
            args.sequence / "rgbs" / f"rgb_{int(source_index):05d}.jpg"
        ).convert("RGB"))
        np.save(frames_dir / f"{observed_index:04d}.npy", image)
    print(json.dumps({
        "sequence": args.sequence.name,
        "observed_rgb_frames": T0 + 1,
        "future_label_frames": len(FRAMES) - T0 - 1,
        "queries": len(ids),
        "future_valid_min": int(validity[T0 + 1 :].sum(axis=1).min()),
        "future_visible_min": int(visibility[T0 + 1 :].sum(axis=1).min()),
        "prediction_uses_gt_depth_or_camera": False,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
