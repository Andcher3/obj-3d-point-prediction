#!/usr/bin/env python3
"""Audit a DAVIS t0-query TrackCraft3R teacher against observed-only caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_davis_joint_probe_bundle import (
    apply_similarity, fit_similarity, graph_audit, project, quantiles, symmetric_knn, transform,
)


def cam0_to_frame_matrices(w2c: np.ndarray) -> np.ndarray:
    return np.einsum("tij,jk->tik", w2c, np.linalg.inv(w2c[0]))


def load_mask(mask_root: Path, video_id: str, frame_index: int, instance_id: int) -> np.ndarray:
    path = mask_root / video_id / f"{frame_index:05d}.png"
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        raise ValueError(f"expected palette/index mask, got RGB at {path}")
    return image == instance_id


def sample_mask(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    xy = np.rint(uv).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    hit = np.zeros(len(xy), dtype=bool)
    hit[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    args = parser.parse_args()
    d = args.video_dir

    manifest = json.loads((d / "query_t0_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((d / "query_frame0_mapping.json").read_text(encoding="utf-8"))
    mf_meta = json.loads((d / "motionforesight_observed_only.json").read_text(encoding="utf-8"))
    query_t0 = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
    query_frame0 = np.asarray(mapping["query_sets"]["48"], dtype=np.float64)
    if query_t0.shape != (48, 2) or query_frame0.shape != (48, 2):
        raise ValueError("expected ordered 48-query manifests")

    with np.load(d / "observed7_user.npz", allow_pickle=True) as obs, \
         np.load(d / "full23_user.npz", allow_pickle=True) as full, \
         np.load(d / "observed7_query_tracks.npz") as observed, \
         np.load(d / "full22_query_tracks.npz") as teacher, \
         np.load(d / "motionforesight_observed_only_arrays.npz") as mf:
        observed_indices = np.asarray(obs["frame_indices"], dtype=np.int32)
        full_indices = np.asarray(full["frame_indices"], dtype=np.int32)
        teacher_indices = np.asarray(teacher["source_frame_indices_output"], dtype=np.int32)
        if not np.array_equal(observed_indices, np.arange(14, 21)):
            raise ValueError(f"observed frame protocol mismatch: {observed_indices.tolist()}")
        if not np.array_equal(full_indices, np.arange(20, 43)):
            raise ValueError(f"teacher input protocol mismatch: {full_indices.tolist()}")
        if not np.array_equal(teacher_indices, np.arange(20, 42)):
            raise ValueError(f"teacher output protocol mismatch: {teacher_indices.tolist()}")
        if not np.allclose(observed["query_uv_original"], query_frame0, atol=1e-4):
            raise ValueError("observed query order differs from verified frame14 mapping")
        if not np.allclose(teacher["query_uv_original"], query_t0, atol=1e-4):
            raise ValueError("teacher query order differs from fixed t0 query order")
        if not np.allclose(mf["query_uv_original"], query_frame0, atol=1e-4):
            raise ValueError("MotionForesight query order differs from verified frame14 mapping")

        observed_world = np.asarray(observed["trackcraft3r_tracks"], dtype=np.float64)
        teacher_cam20 = np.asarray(teacher["trackcraft3r_tracks"], dtype=np.float64)
        if observed_world.shape != (7, 48, 3) or teacher_cam20.shape != (22, 48, 3):
            raise ValueError(f"unexpected track shapes: {observed_world.shape}, {teacher_cam20.shape}")

        obs_w2c = np.asarray(obs["extrinsics_w2c"], dtype=np.float64)
        full_w2c = np.asarray(full["extrinsics_w2c"], dtype=np.float64)
        # Dense observed TrackCraft output is in the DA3 world frame used by
        # observed7_user.npz, so its last-camera conversion uses W2C directly.
        observed_cam20_all = transform(obs_w2c[-1], observed_world)
        observed_t0 = observed_cam20_all[-1]
        teacher_t0 = teacher_cam20[0]
        obs_k = np.asarray(obs["fx_fy_cx_cy"], dtype=np.float64)
        full_k = np.asarray(full["fx_fy_cx_cy"], dtype=np.float64)
        observed_uv_t0 = project(observed_t0, obs_k)
        teacher_uv_t0 = project(teacher_t0, full_k)
        bbox_2d = max(float(manifest["bbox_diagonal_px"]), 1e-8)
        observed_anchor = np.linalg.norm(observed_uv_t0 - query_t0, axis=-1) / bbox_2d
        teacher_anchor = np.linalg.norm(teacher_uv_t0 - query_t0, axis=-1) / bbox_2d
        cross_context_uv = np.linalg.norm(observed_uv_t0 - teacher_uv_t0, axis=-1) / bbox_2d

        fold_error = np.full(48, np.nan, dtype=np.float64)
        folds = []
        for fold in range(4):
            held = np.arange(48) % 4 == fold
            fit = fit_similarity(teacher_t0[~held], observed_t0[~held])
            fold_error[held] = np.linalg.norm(
                apply_similarity(teacher_t0[held], fit) - observed_t0[held], axis=-1
            )
            folds.append({
                "fold": fold, "train_queries": int((~held).sum()),
                "held_queries": int(held.sum()), "held_error_m": quantiles(fold_error[held]),
            })
        bbox_3d = max(float(np.linalg.norm(np.ptp(observed_t0, axis=0))), 1e-8)
        registration = fold_error / bbox_3d
        full_fit = fit_similarity(teacher_t0, observed_t0)
        teacher_aligned = apply_similarity(teacher_cam20, full_fit)
        future = teacher_aligned[1:10]

        observed_valid = (
            np.isfinite(observed_cam20_all).all(-1).all(0)
            & (observed_cam20_all[..., 2] > 0).all(0)
        )
        teacher_valid = np.isfinite(teacher_cam20[1:10]).all(-1).all(0) & (teacher_cam20[1:10, :, 2] > 0).all(0)
        valid = observed_valid & teacher_valid & np.asarray(observed["oob_mask"], bool) & np.asarray(teacher["oob_mask"], bool)
        graph = graph_audit(48, symmetric_knn(query_t0, 3), valid)

        full_cam = cam0_to_frame_matrices(full_w2c)
        future_uv = np.stack([
            project(transform(full_cam[j], teacher_cam20[j]), full_k) for j in range(1, 10)
        ])
        instance_id = int(manifest["png_palette_index"])
        mask_hits = []
        for slot, frame_index in enumerate(range(21, 30)):
            hit = sample_mask(load_mask(args.mask_root, manifest["video_id"], frame_index, instance_id), future_uv[slot])
            mask_hits.append({"source_frame_index": frame_index, "hit_count": int(hit.sum()), "fraction": float(hit.mean())})

        p0_all = np.asarray(mf["p0_query"], dtype=np.float32)
        h_all = np.asarray(mf["h_output_latent"], dtype=np.float32)
        if p0_all.shape[0] < 16 or h_all.shape[0] < 16 or p0_all.shape[1:] != (48, 3) or h_all.shape[1:2] != (48,):
            raise ValueError(f"unexpected MotionForesight arrays: {p0_all.shape}, {h_all.shape}")
        p0 = p0_all[7:16]
        h_output = h_all[7:16]
        future_frames_supplied = bool(mf_meta.get("input", {}).get("future_frame_tensor_supplied", True))
        gates = {
            "query_count_48": True,
            "teacher_query_order_matches_t0": True,
            "motionforesight_query_order_matches_observed_mapping": True,
            "future_frames_not_supplied_to_model": not future_frames_supplied,
            "observed_t0_anchor_p90_le_5pct": float(np.quantile(observed_anchor, .9)) <= .05,
            "observed_t0_anchor_max_le_10pct": float(observed_anchor.max()) <= .10,
            "teacher_t0_anchor_p90_le_5pct": float(np.quantile(teacher_anchor, .9)) <= .05,
            "teacher_t0_anchor_max_le_10pct": float(teacher_anchor.max()) <= .10,
            "valid_queries_at_least_44": int(valid.sum()) >= 44,
            "registration_p90_le_10pct": float(np.quantile(registration, .9)) <= .10,
            "registration_90pct_below_10pct": float((registration <= .10).mean()) >= .90,
            "registration_max_le_20pct": float(registration.max()) <= .20,
            "valid_graph_connected": bool(graph["connected"]),
            "finite_baseline_teacher": bool(np.isfinite(p0).all() and np.isfinite(h_output).all() and np.isfinite(future).all()),
            "official_teacher_interleaved_12_frame_runs": bool(teacher["official_long_video_protocol"]),
        }
        payload = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "training_approved": False,
            "video_id": manifest["video_id"],
            "protocol": "davis_t0_query_official_interleaved23_2026-09-29",
            "observed_source_frames": observed_indices.tolist(),
            "offline_teacher_input_source_frames": full_indices.tolist(),
            "teacher_output_source_frames": teacher_indices.tolist(),
            "evaluated_future_source_frames": list(range(21, 30)),
            "preview_mp4_fps_used": False,
            "time_basis": "consecutive DAVIS source frame indices; capture fps unavailable",
            "gates": gates,
            "valid_query_count": int(valid.sum()),
            "observed_t0_anchor_normalized_bbox": quantiles(observed_anchor),
            "teacher_t0_anchor_normalized_bbox": quantiles(teacher_anchor),
            "cross_context_t0_uv_normalized_bbox": quantiles(cross_context_uv),
            "registration_crossfit_normalized_3d_bbox": quantiles(registration),
            "registration_fraction_below_10pct": float((registration <= .10).mean()),
            "registration_folds": folds,
            "graph": graph,
            "future_mask_hits": mask_hits,
            "shapes": {"p0": list(p0.shape), "h_output": list(h_output.shape), "teacher_future": list(future.shape)},
            "interpretation_limit": "Future mask hits are diagnostics and are not an unregistered post-hoc gate.",
        }
        np.savez_compressed(
            d / "p0_h_cache_t0_teacher.npz",
            p0_formal9=p0, h_output_latent_formal9=h_output,
            pseudo_gt_aligned=future.astype(np.float32),
            pseudo_gt_t0_aligned=apply_similarity(teacher_t0, full_fit).astype(np.float32),
            pseudo_gt_validity=np.broadcast_to(valid, (9, 48)),
            query_anchor_match_error_px=(teacher_anchor * bbox_2d).astype(np.float32),
            query_uv_t0_source=query_t0.astype(np.float32),
            observed_only_trackcraft_t0=observed_t0.astype(np.float32),
            registration_crossfit_error_normalized=registration.astype(np.float32),
            coordinate_frame=np.asarray("observed_only_last_camera_source_frame20"),
            prediction_source=np.asarray("motionforesight_observed_only_7_frames"),
            pseudo_gt_source=np.asarray("trackcraft3r_t0_query_offline_interleaved23"),
            future_frames_supplied_to_model=np.asarray(False),
        )

    (d / "quality_audit_t0_teacher.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
