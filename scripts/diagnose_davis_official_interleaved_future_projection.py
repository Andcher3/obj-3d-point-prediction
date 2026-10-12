#!/usr/bin/env python3
"""Diagnose indexing, camera direction, and mask hits for official interleaved tracks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def transform(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate((xyz, np.ones((*xyz.shape[:-1], 1))), axis=-1)
    return np.einsum("tij,tnj->tni", matrix, xyz1)[..., :3]


def project(points: np.ndarray, matrices: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = transform(matrices, points)
    z = camera[..., 2]
    uv = np.stack((k[0] * camera[..., 0] / z + k[2], k[1] * camera[..., 1] / z + k[3]), axis=-1)
    return uv, np.isfinite(uv).all(-1) & np.isfinite(z) & (z > 0)


def matrices_for_mode(w2c: np.ndarray, mode: str) -> np.ndarray:
    if mode == "w2c_per_frame":
        return np.einsum("tij,jk->tik", w2c, np.linalg.inv(w2c[0]))
    if mode == "inverse_as_w2c":
        return np.linalg.inv(w2c)
    if mode == "identity":
        return np.broadcast_to(np.eye(4), w2c.shape).copy()
    if mode == "w2c0_fixed":
        return np.broadcast_to(w2c[0], w2c.shape).copy()
    raise ValueError(mode)


def mask_hits(uv: np.ndarray, valid: np.ndarray, source_frames: np.ndarray, processed_hw: tuple[int, int], mask_dir: Path) -> list[int]:
    ph, pw = processed_hw
    result = []
    for frame_uv, frame_valid, source_index in zip(uv, valid, source_frames):
        mask = np.asarray(Image.open(mask_dir / f"{int(source_index):05d}.png"), dtype=np.uint8) == 1
        h, w = mask.shape
        xy = np.rint(frame_uv * np.asarray([w / pw, h / ph])).astype(np.int64)
        inside = frame_valid & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        result.append(int(np.sum(mask[xy[inside, 1], xy[inside, 0]])))
    return result


def projection_diagnostic(points: np.ndarray, w2c: np.ndarray, k: np.ndarray, frames: np.ndarray, processed_hw: tuple[int, int], mask_dir: Path) -> dict:
    def finite_xy_summary(uv: np.ndarray, reducer) -> list[list[float] | None]:
        result = []
        for frame_uv in uv:
            finite = frame_uv[np.isfinite(frame_uv).all(axis=-1)]
            result.append(None if len(finite) == 0 else reducer(finite, axis=0).astype(float).tolist())
        return result

    output = {}
    for mode in ("w2c_per_frame", "inverse_as_w2c", "identity", "w2c0_fixed"):
        uv, valid = project(points, matrices_for_mode(w2c, mode), k)
        output[mode] = {
            "positive_finite_count": np.sum(valid, axis=1).astype(int).tolist(),
            "mask_hits_of_48": mask_hits(uv, valid, frames, processed_hw, mask_dir),
            "uv_p50_xy": finite_xy_summary(uv, lambda x, axis: np.quantile(x, 0.5, axis=axis)),
            "uv_min_xy": finite_xy_summary(uv, np.min),
            "uv_max_xy": finite_xy_summary(uv, np.max),
        }
    return output


def pointmap_coordinate_roundtrip(depth: np.ndarray, w2c: np.ndarray, k: np.ndarray) -> dict:
    """Verify the predictor's camera_j -> camera_0 pointmap convention."""
    h, w = depth.shape[1:]
    vv, uu = np.meshgrid(np.arange(0, h, 8), np.arange(0, w, 8), indexing="ij")
    target_uv = np.stack((uu, vv), axis=-1).reshape(-1, 2).astype(np.float64)
    c2w = np.linalg.inv(w2c)
    w2c0 = w2c[0]
    c2w0 = np.linalg.inv(w2c0)
    roundtrip, identity_error = [], []
    fx, fy, cx, cy = k
    for j in range(len(depth)):
        z = depth[j, vv, uu].reshape(-1).astype(np.float64)
        camj = np.stack(((target_uv[:, 0] - cx) / fx * z, (target_uv[:, 1] - cy) / fy * z, z), axis=-1)
        camj1 = np.concatenate((camj, np.ones((len(camj), 1))), axis=-1)
        cam0 = (camj1 @ c2w[j].T) @ w2c0.T
        cam0 = cam0[:, :3]
        cam01 = np.concatenate((cam0, np.ones((len(cam0), 1))), axis=-1)
        recovered = (cam01 @ c2w0.T) @ w2c[j].T
        recovered = recovered[:, :3]
        uv = np.stack((fx * recovered[:, 0] / recovered[:, 2] + cx, fy * recovered[:, 1] / recovered[:, 2] + cy), axis=-1)
        uv_identity = np.stack((fx * cam0[:, 0] / cam0[:, 2] + cx, fy * cam0[:, 1] / cam0[:, 2] + cy), axis=-1)
        roundtrip.append(float(np.max(np.linalg.norm(uv - target_uv, axis=-1))))
        identity_error.append(float(np.quantile(np.linalg.norm(uv_identity - target_uv, axis=-1), 0.9)))
    return {
        "convention": "Pj is camera_j depth unprojected and transformed into frame0 camera; projection to frame j uses w2c_j @ c2w_0",
        "sample_stride_px": 8,
        "roundtrip_max_error_px_per_frame": roundtrip,
        "identity_projection_p90_error_px_per_frame": identity_error,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-dir", type=Path, required=True)
    p.add_argument("--legacy-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--trackcraft-repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    with np.load(a.official_dir / "full23_user.npz", allow_pickle=True) as user, \
         np.load(a.official_dir / "full22_query_tracks.npz") as tracks, \
         np.load(a.legacy_dir / "full22_user.npz", allow_pickle=True) as legacy_user, \
         np.load(a.legacy_dir / "full22_query_tracks.npz") as legacy_tracks:
        w2c = np.asarray(user["extrinsics_w2c"], dtype=np.float64)
        k = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
        processed_hw = tuple(int(x) for x in user["depth_map"].shape[1:3])
        run_indices = np.asarray(tracks["interleaved_run_indices"], dtype=np.int64)
        source_input = np.asarray(tracks["source_frame_indices_input"], dtype=np.int64)
        source_output = np.asarray(tracks["source_frame_indices_output"], dtype=np.int64)
        raw_runs = np.asarray(tracks["trackcraft3r_tracks_raw_runs"], dtype=np.float64)
        anchored_runs = np.asarray(tracks["trackcraft3r_tracks_anchored_runs"], dtype=np.float64)
        raw_merged = np.asarray(tracks["trackcraft3r_tracks_raw_interleaved"], dtype=np.float64)
        anchored_merged = np.asarray(tracks["trackcraft3r_tracks"], dtype=np.float64)
        legacy_w2c = np.asarray(legacy_user["extrinsics_w2c"], dtype=np.float64)
        legacy_k = np.asarray(legacy_user["fx_fy_cx_cy"], dtype=np.float64)
        legacy_points = np.asarray(legacy_tracks["trackcraft3r_tracks"], dtype=np.float64)
        stored = {key: user[key] for key in user.files}

    sys.path.insert(0, str(a.trackcraft_repo))
    import evaluation.dust3r_eval_utils as dust3r_utils
    stored["tracks_XYZ"] = np.zeros((23, 1, 3), dtype=np.float32)
    stored["visibility"] = np.ones((23, 1), dtype=bool)
    fd, temp = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    np.savez_compressed(temp, **stored)
    try:
        _, _, _, loaded_k, _, _, _, loaded_w2c = dust3r_utils.load_npz_data(temp, num_frames=23)
    finally:
        os.unlink(temp)
    loaded_w2c = np.asarray(loaded_w2c, dtype=np.float64)
    loaded_k = np.asarray(loaded_k, dtype=np.float64)

    runs = {}
    for run_id, indices in enumerate(run_indices):
        frames = source_input[indices]
        run_w2c = w2c[indices]
        runs[str(run_id)] = {
            "global_indices": indices.tolist(),
            "source_frame_indices": frames.tolist(),
            "w2c_translation_norm": np.linalg.norm(run_w2c[:, :3, 3], axis=-1).tolist(),
            "raw": projection_diagnostic(raw_runs[run_id], run_w2c, k, frames, processed_hw, a.mask_dir),
            "anchored": projection_diagnostic(anchored_runs[run_id], run_w2c, k, frames, processed_hw, a.mask_dir),
            "temporal_displacement_max_abs_change": float(np.max(np.abs(
                (anchored_runs[run_id] - anchored_runs[run_id, 0:1]) -
                (raw_runs[run_id] - raw_runs[run_id, 0:1])
            ))),
        }

    output = {
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING",
        "video_id": a.official_dir.name,
        "official_run_indices": run_indices.tolist(),
        "official_source_frame_indices_input": source_input.tolist(),
        "official_source_frame_indices_output": source_output.tolist(),
        "load_npz_data_w2c_max_abs_difference_from_stored": float(np.max(np.abs(loaded_w2c - w2c))),
        "load_npz_data_intrinsics_max_abs_difference_from_stored": float(np.max(np.abs(loaded_k - k))),
        "stored_frame0_w2c_max_abs_difference_identity": float(np.max(np.abs(w2c[0] - np.eye(4)))),
        "full23_vs_legacy22_w2c_first22_max_abs_difference": float(np.max(np.abs(w2c[:22] - legacy_w2c))),
        "full23_vs_legacy22_intrinsics_max_abs_difference": float(np.max(np.abs(k - legacy_k))),
        "predictor_pointmap_coordinate_roundtrip": pointmap_coordinate_roundtrip(
            np.asarray(stored["depth_map"], dtype=np.float64), w2c, k
        ),
        "runs_before_merge": runs,
        "official_merged_raw": projection_diagnostic(raw_merged, w2c[:22], k, source_output, processed_hw, a.mask_dir),
        "official_merged_anchored": projection_diagnostic(anchored_merged, w2c[:22], k, source_output, processed_hw, a.mask_dir),
        "legacy_direct22_separate_context": projection_diagnostic(legacy_points, legacy_w2c, legacy_k, source_output, processed_hw, a.mask_dir),
        "interpretation_limit": "Projection alternatives diagnose indexing/pose conventions only; high mask hits do not validate 3D teacher quality.",
    }
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "video_id": output["video_id"],
        "load_w2c_difference": output["load_npz_data_w2c_max_abs_difference_from_stored"],
        "official_w2c_hits": output["official_merged_anchored"]["w2c_per_frame"]["mask_hits_of_48"],
        "official_identity_hits": output["official_merged_anchored"]["identity"]["mask_hits_of_48"],
        "legacy_w2c_hits": output["legacy_direct22_separate_context"]["w2c_per_frame"]["mask_hits_of_48"],
    }, indent=2))


if __name__ == "__main__":
    main()
