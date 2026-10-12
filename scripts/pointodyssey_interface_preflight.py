#!/usr/bin/env python3
"""Audit a PointOdyssey 7+15 window without using future labels for query choice.

This is a data-interface pilot, not a training or benchmark runner.  Target
selection and query sampling use only the t0 mask and observed trajectories.
Future annotations enter only the post-selection validity audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


FRAMES = np.arange(0, 44, 2, dtype=np.int64)
T0 = 6
N_QUERY = 48


def farthest_points(xy: np.ndarray, n: int) -> np.ndarray:
    center = np.median(xy, axis=0)
    first = int(np.argmin(np.sum((xy - center) ** 2, axis=1)))
    selected = [first]
    nearest = np.sum((xy - xy[first]) ** 2, axis=1)
    for _ in range(1, n):
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        nearest = np.minimum(nearest, np.sum((xy - xy[nxt]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


def audit_sequence(seq: Path, target_ids: list[int] | None = None) -> dict:
    anno = np.load(seq / "anno.npz")
    t0_frame = int(FRAMES[T0])
    mask = np.asarray(Image.open(seq / "masks" / f"mask_{t0_frame:05d}.png"))
    height, width = mask.shape[:2]
    tracks = anno["trajs_2d"][FRAMES]
    xyz_world_all = anno["trajs_3d"][FRAMES]
    xyz_world = xyz_world_all[T0]
    valid = anno["valids"][FRAMES].astype(bool)
    visible = anno["visibs"][FRAMES].astype(bool)
    uv = tracks[T0]
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz_world).all(axis=1)
    x = np.rint(np.where(finite, uv[:, 0], -1)).astype(np.int64)
    y = np.rint(np.where(finite, uv[:, 1], -1)).astype(np.int64)
    in_frame = finite & (x >= 1) & (x < width - 1) & (y >= 1) & (y < height - 1)
    labels = np.zeros(len(uv), dtype=mask.dtype)
    labels[in_frame] = mask[y[in_frame], x[in_frame]]
    inside = np.zeros(len(uv), dtype=bool)
    inside[in_frame] = (
        (mask[y[in_frame] - 1, x[in_frame]] == labels[in_frame])
        & (mask[y[in_frame] + 1, x[in_frame]] == labels[in_frame])
        & (mask[y[in_frame], x[in_frame] - 1] == labels[in_frame])
        & (mask[y[in_frame], x[in_frame] + 1] == labels[in_frame])
    )
    t0_good = in_frame & inside & valid[T0] & visible[T0]

    candidate_rows = []
    for label, pixels in zip(*np.unique(mask, return_counts=True)):
        label = int(label)
        area_fraction = float(pixels / mask.size)
        ids = np.flatnonzero(t0_good & (labels == label))
        if label == 0 or not (0.002 <= area_fraction <= 0.35) or len(ids) < N_QUERY:
            continue
        candidate_rows.append({
            "mask_id": label,
            "area_fraction_t0": area_fraction,
            "t0_track_count": int(len(ids)),
            "observed_median_displacement_px": float(np.median(
                np.linalg.norm(tracks[T0, ids] - tracks[0, ids], axis=1)
            )),
        })
    candidate_rows.sort(key=lambda row: (-row["t0_track_count"], row["mask_id"]))

    selected_ids = target_ids if target_ids is not None else ([candidate_rows[0]["mask_id"]] if candidate_rows else [])
    target_track_ids = np.flatnonzero(t0_good & np.isin(labels, selected_ids))
    result = {
        "sequence": seq.name,
        "frame_indices": FRAMES.tolist(),
        "t0_window_index": T0,
        "t0_source_frame": t0_frame,
        "fps_assumption": 30,
        "model_fps": 15,
        "image_shape_hw": [height, width],
        "candidate_masks": candidate_rows,
        "target_selection_source": "manual_t0_object_grouping" if target_ids is not None else "maximum_t0_track_support_single_mask",
        "selected_mask_ids": selected_ids,
        "status": "FAIL_NO_TARGET" if len(target_track_ids) < N_QUERY else "PENDING",
    }
    if len(target_track_ids) < N_QUERY:
        return result

    chosen_ids = target_track_ids[farthest_points(uv[target_track_ids], N_QUERY)]
    camera = anno["extrinsics"][t0_frame].astype(np.float64)
    intrinsics = anno["intrinsics"][t0_frame].astype(np.float64)
    homogeneous = np.c_[xyz_world[chosen_ids], np.ones(N_QUERY)]
    xyz_camera = (camera @ homogeneous.T).T[:, :3]
    uvh = (intrinsics @ xyz_camera.T).T
    uv_projected = uvh[:, :2] / uvh[:, 2:3]
    projection_error = np.linalg.norm(uv_projected - uv[chosen_ids], axis=1)
    all_world = xyz_world_all[:, chosen_ids].astype(np.float64)
    all_cameras = anno["extrinsics"][FRAMES].astype(np.float64)
    all_intrinsics = anno["intrinsics"][FRAMES].astype(np.float64)
    all_camera_xyz = np.einsum(
        "tij,tnj->tni", all_cameras[:, :3, :3], all_world
    ) + all_cameras[:, None, :3, 3]
    all_uvh = np.einsum("tij,tnj->tni", all_intrinsics, all_camera_xyz)
    all_projected = all_uvh[..., :2] / all_uvh[..., 2:3]
    all_projection_error = np.linalg.norm(all_projected - tracks[:, chosen_ids], axis=-1)
    all_usable = valid[:, chosen_ids] & visible[:, chosen_ids] & (all_camera_xyz[..., 2] > 0)
    future_projection_p90 = float(np.quantile(all_projection_error[T0 + 1 :][all_usable[T0 + 1 :]], 0.9))
    future_good = valid[T0 + 1 :, chosen_ids] & np.isfinite(all_world[T0 + 1 :]).all(axis=-1)
    per_step = future_good.sum(axis=1)
    visible_per_step = (future_good & visible[T0 + 1 :, chosen_ids]).sum(axis=1)
    gate = bool(
        np.isfinite(projection_error).all()
        and np.all(xyz_camera[:, 2] > 0)
        and np.quantile(projection_error, 0.9) <= 2.0
        and future_projection_p90 <= 2.0
        and np.all(per_step >= 32)
    )
    result.update({
        "status": "PASS" if gate else "FAIL",
        "selected_mask_id": selected_ids[0] if len(selected_ids) == 1 else None,
        "selected_t0_track_pool_count": int(len(target_track_ids)),
        "query_point_indices": chosen_ids.tolist(),
        "query_uv_t0": uv[chosen_ids].tolist(),
        "projection_error_p90_px": float(np.quantile(projection_error, 0.9)),
        "projection_error_max_px": float(projection_error.max()),
        "future_projection_error_p90_px": future_projection_p90,
        "future_gt_valid_per_step": per_step.tolist(),
        "future_gt_valid_min": int(per_step.min()),
        "future_visible_valid_per_step": visible_per_step.tolist(),
        "future_visible_valid_min": int(visible_per_step.min()),
        "t0_gt_bbox_diagonal_camera": float(np.linalg.norm(np.ptp(xyz_camera, axis=0))),
        "checks": {
            "projection_p90_le_2px": bool(np.quantile(projection_error, 0.9) <= 2.0),
            "future_projection_p90_le_2px": bool(future_projection_p90 <= 2.0),
            "future_each_step_ge_32": bool(np.all(per_step >= 32)),
            "positive_camera_depth": bool(np.all(xyz_camera[:, 2] > 0)),
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, nargs="+", required=True)
    parser.add_argument("--target-specs-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.target_specs_json.read_text(encoding="utf-8"))["targets"] if args.target_specs_json else {}
    rows = [audit_sequence(path, specs[path.name]["mask_ids"] if path.name in specs else None) for path in args.sequence]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sequences": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(row["sequence"], row["status"], row.get("selected_mask_id"), row.get("projection_error_p90_px"), row.get("future_visible_valid_min"))


if __name__ == "__main__":
    main()
