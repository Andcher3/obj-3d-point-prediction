#!/usr/bin/env python3
"""Audit a direct frame-0 query teacher against observed-only trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_davis_joint_probe_bundle import (
    apply_similarity, fit_similarity, graph_audit, project, quantiles,
    symmetric_knn, transform,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    args = ap.parse_args()
    d = args.video_dir
    manifest = json.loads((d / "query_frame0_manifest.json").read_text(encoding="utf-8"))
    query = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
    if query.shape != (48, 2) or manifest["query_reference_source_frame_index"] != 14:
        raise ValueError("invalid direct frame-0 manifest")
    with (np.load(d / "observed7_user.npz", allow_pickle=True) as obs,
          np.load(d / "full22_user.npz", allow_pickle=True) as full,
          np.load(d / "observed7_query_tracks.npz") as observed,
          np.load(d / "full22_query_tracks.npz") as teacher,
          np.load(d / "motionforesight_observed_only_arrays.npz") as mf):
        observed_indices = np.asarray(obs["frame_indices"]).tolist()
        full_indices = np.asarray(full["frame_indices"]).tolist()
        if observed_indices != list(range(14, 21)) or full_indices != list(range(14, 36)):
            raise ValueError("source frame order mismatch")
        if not np.allclose(teacher["query_uv_original"], query, atol=1e-4):
            raise ValueError("full teacher query order differs from frame-0 manifest")
        if not np.allclose(mf["query_uv_original"], query, atol=1e-4):
            raise ValueError("MotionForesight query order differs from frame-0 manifest")
        if not np.allclose(observed["query_uv_original"], query, atol=1e-4):
            raise ValueError("observed query order differs from frame-0 manifest")
        observed_world = np.asarray(observed["trackcraft3r_tracks"], dtype=np.float64)
        if observed_world.shape != (7, 48, 3):
            raise ValueError(f"expected observed (7,48,3), got {observed_world.shape}")
        teacher_world = np.asarray(teacher["trackcraft3r_tracks"], dtype=np.float64)
        if teacher_world.shape != (22, 48, 3):
            raise ValueError(f"expected (22,48,3), got {teacher_world.shape}")
        obs_w2c = np.asarray(obs["extrinsics_w2c"], dtype=np.float64)
        full_w2c = np.asarray(full["extrinsics_w2c"], dtype=np.float64)
        observed_cam20 = transform(obs_w2c[6], observed_world)
        teacher_cam20 = transform(full_w2c[6], teacher_world)
        obs_t0, teacher_t0 = observed_cam20[6], teacher_cam20[6]
        obs_intr = np.asarray(obs["fx_fy_cx_cy"], dtype=np.float64)
        full_intr = np.asarray(full["fx_fy_cx_cy"], dtype=np.float64)
        observed_uv_t0 = project(obs_t0, obs_intr)
        teacher_uv_t0 = project(teacher_t0, full_intr)
        obs_uv_0 = project(transform(obs_w2c[0], observed_world[0]), obs_intr)
        teacher_uv_0 = project(transform(full_w2c[0], teacher_world[0]), full_intr)
        bbox_diag = max(float(manifest["bbox_diagonal_px"]), 1e-8)
        anchor_obs = np.linalg.norm(obs_uv_0 - query, axis=-1) / bbox_diag
        anchor_teacher = np.linalg.norm(teacher_uv_0 - query, axis=-1) / bbox_diag
        pair_uv = np.linalg.norm(observed_uv_t0 - teacher_uv_t0, axis=-1) / bbox_diag
        # The target at frame 20 is unknown to the query sampler; this is a
        # cross-context t0 consistency check, not an assumed t0 UV identity.
        fold_error = np.full(48, np.nan, dtype=np.float64)
        for fold in range(4):
            held = np.arange(48) % 4 == fold
            fit = fit_similarity(teacher_t0[~held], obs_t0[~held])
            fold_error[held] = np.linalg.norm(apply_similarity(teacher_t0[held], fit) - obs_t0[held], axis=-1)
        bbox_3d = max(float(np.linalg.norm(np.ptp(obs_t0, axis=0))), 1e-8)
        registration = fold_error / bbox_3d
        fit = fit_similarity(teacher_t0, obs_t0)
        aligned = apply_similarity(teacher_cam20, fit)
        future = aligned[7:16]
        observed_valid = np.isfinite(observed_cam20).all(-1).all(0) & (observed_cam20[..., 2] > 0).all(0)
        teacher_future_valid = np.isfinite(teacher_cam20[7:16]).all(-1).all(0) & (teacher_cam20[7:16, :, 2] > 0).all(0)
        oob = np.asarray(teacher["oob_mask"], dtype=bool) & np.asarray(observed["oob_mask"], dtype=bool)
        valid = observed_valid & teacher_future_valid & oob
        graph = graph_audit(48, symmetric_knn(observed_uv_t0, 3), valid)
        p0_all = np.asarray(mf["p0_query"], dtype=np.float32)
        h_all = np.asarray(mf["h_output_latent"], dtype=np.float32)
        if p0_all.shape != (22, 48, 3) or h_all.shape[:2] != (22, 48):
            raise ValueError(f"unexpected MotionForesight P0/H shape: {p0_all.shape}, {h_all.shape}")
        if bool(np.asarray(mf["future_frames_supplied_to_model"])):
            raise ValueError("future frames entered MotionForesight inference")
        p0, h_output = p0_all[7:16], h_all[7:16]
        gates = {
            "frame0_query_48": True,
            "observed_frame0_anchor_p90_le_5pct": float(np.quantile(anchor_obs, .9)) <= .05,
            "observed_frame0_anchor_max_le_10pct": float(anchor_obs.max()) <= .10,
            "teacher_frame0_anchor_p90_le_5pct": float(np.quantile(anchor_teacher, .9)) <= .05,
            "teacher_frame0_anchor_max_le_10pct": float(anchor_teacher.max()) <= .10,
            "cross_context_t0_uv_p90_le_5pct": float(np.quantile(pair_uv, .9)) <= .05,
            "cross_context_t0_uv_max_le_10pct": float(pair_uv.max()) <= .10,
            "valid_queries_at_least_44": int(valid.sum()) >= 44,
            "registration_p90_le_10pct": float(np.quantile(registration, .9)) <= .10,
            "registration_90pct_below_10pct": float((registration <= .10).mean()) >= .90,
            "registration_max_le_20pct": float(registration.max()) <= .20,
            "valid_graph_connected": bool(graph["connected"]),
            "finite_predictions_teacher": bool(np.isfinite(p0).all() and np.isfinite(h_output).all() and np.isfinite(future).all()),
        }
        payload = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "video_id": manifest["video_id"],
            "protocol": "frame0_direct_query_2026-09-28",
            "query_reference_source_frame_index": 14,
            "output_camera_source_frame_index": 20,
            "source_frame_indices": full_indices,
            "observed_source_frame_indices": observed_indices,
            "evaluated_future_source_frame_indices": list(range(21, 30)),
            "preview_mp4_fps_used": False,
            "gates": gates,
            "valid_query_count": int(valid.sum()),
            "observed_frame0_anchor_normalized_bbox": quantiles(anchor_obs),
            "teacher_frame0_anchor_normalized_bbox": quantiles(anchor_teacher),
            "cross_context_t0_uv_normalized_frame0_bbox": quantiles(pair_uv),
            "registration_crossfit_normalized_3d_bbox": quantiles(registration),
            "registration_fraction_below_10pct": float((registration <= .10).mean()),
            "graph": graph,
            "shapes": {"observed": list(observed_world.shape), "teacher": list(teacher_world.shape), "p0": list(p0.shape), "h_output": list(h_output.shape)},
            "sources": {"baseline": "MotionForesight, seven observed frames only", "pseudo_gt": "TrackCraft3R, offline full22 query-only"},
        }
        np.savez_compressed(
            d / "p0_h_cache_aligned.npz",
            p0_formal9=p0, h_output_latent_formal9=h_output,
            pseudo_gt_aligned=future.astype(np.float32),
            pseudo_gt_t0_aligned=aligned[6].astype(np.float32),
            pseudo_gt_validity=np.broadcast_to(valid, (9, 48)),
            query_anchor_match_error_px=(anchor_teacher * bbox_diag).astype(np.float32),
            observed_only_trackcraft_t0=obs_t0.astype(np.float32),
            query_uv_t0_source=observed_uv_t0.astype(np.float32),
            query_uv_frame0_input=query.astype(np.float32),
            registration_crossfit_error_normalized=registration.astype(np.float32),
            coordinate_frame=np.asarray("observed_only_last_camera_source_frame20"),
            query_reference_frame=np.asarray("source_frame14"),
            prediction_source=np.asarray("motionforesight_observed_only"),
            pseudo_gt_source=np.asarray("trackcraft3r_offline_full22_query_only"),
            time_basis=np.asarray("consecutive_davis_source_frame_index"),
        )
    (d / "quality_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
