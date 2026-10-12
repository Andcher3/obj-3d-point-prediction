#!/usr/bin/env python3
"""Apply the fixed DAVIS quality gates to official interleaved Pj0-anchored tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_davis_joint_probe_bundle import (
    apply_similarity, fit_similarity, graph_audit, project, quantiles, symmetric_knn, transform,
)


def cam0_to_frame_matrices(w2c: np.ndarray) -> np.ndarray:
    return np.einsum("tij,jk->tik", w2c, np.linalg.inv(w2c[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    args = parser.parse_args(); d = args.video_dir
    manifest = json.loads((d / "query_frame0_manifest.json").read_text(encoding="utf-8"))
    query = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
    with np.load(d / "observed7_user.npz", allow_pickle=True) as obs, \
         np.load(d / "full23_user.npz", allow_pickle=True) as full, \
         np.load(d / "observed7_query_tracks.npz") as observed, \
         np.load(d / "full22_query_tracks.npz") as teacher, \
         np.load(d / "motionforesight_observed_only_arrays.npz") as mf:
        observed_indices = np.asarray(obs["frame_indices"]).tolist()
        full_input_indices = np.asarray(full["frame_indices"]).tolist()
        teacher_indices = list(range(14, 36))
        if observed_indices != list(range(14, 21)) or full_input_indices != list(range(14, 37)):
            raise ValueError("source frame protocol mismatch")
        observed_world = np.asarray(observed["trackcraft3r_tracks"], dtype=np.float64)
        teacher_world = np.asarray(teacher["trackcraft3r_tracks"], dtype=np.float64)
        if observed_world.shape != (7, 48, 3) or teacher_world.shape != (22, 48, 3):
            raise ValueError("query track shape mismatch")
        if not np.allclose(observed["query_uv_original"], query, atol=1e-4) or not np.allclose(teacher["query_uv_original"], query, atol=1e-4):
            raise ValueError("query order mismatch")
        obs_w2c = np.asarray(obs["extrinsics_w2c"], dtype=np.float64)
        full_w2c = np.asarray(full["extrinsics_w2c"], dtype=np.float64)[:22]
        obs_cam = cam0_to_frame_matrices(obs_w2c)
        full_cam = cam0_to_frame_matrices(full_w2c)
        observed_cam20 = transform(obs_cam[6], observed_world)
        teacher_cam20 = transform(full_cam[6], teacher_world)
        obs_t0, teacher_t0 = observed_cam20[6], teacher_cam20[6]
        obs_k = np.asarray(obs["fx_fy_cx_cy"], dtype=np.float64)
        full_k = np.asarray(full["fx_fy_cx_cy"], dtype=np.float64)
        observed_uv_t0 = project(obs_t0, obs_k); teacher_uv_t0 = project(teacher_t0, full_k)
        obs_uv0 = project(transform(obs_cam[0], observed_world[0]), obs_k)
        teacher_uv0 = project(transform(full_cam[0], teacher_world[0]), full_k)
        bbox_diag = max(float(manifest["bbox_diagonal_px"]), 1e-8)
        anchor_obs = np.linalg.norm(obs_uv0 - query, axis=-1) / bbox_diag
        anchor_teacher = np.linalg.norm(teacher_uv0 - query, axis=-1) / bbox_diag
        pair_uv = np.linalg.norm(observed_uv_t0 - teacher_uv_t0, axis=-1) / bbox_diag
        fold_error = np.full(48, np.nan, dtype=np.float64)
        for fold in range(4):
            held = np.arange(48) % 4 == fold
            fit = fit_similarity(teacher_t0[~held], obs_t0[~held])
            fold_error[held] = np.linalg.norm(apply_similarity(teacher_t0[held], fit) - obs_t0[held], axis=-1)
        bbox_3d = max(float(np.linalg.norm(np.ptp(obs_t0, axis=0))), 1e-8)
        registration = fold_error / bbox_3d
        fit = fit_similarity(teacher_t0, obs_t0)
        aligned = apply_similarity(teacher_cam20, fit); future = aligned[7:16]
        observed_valid = np.isfinite(observed_cam20).all(-1).all(0) & (observed_cam20[..., 2] > 0).all(0)
        teacher_valid = np.isfinite(teacher_cam20[7:16]).all(-1).all(0) & (teacher_cam20[7:16, :, 2] > 0).all(0)
        valid = observed_valid & teacher_valid & np.asarray(teacher["oob_mask"], bool) & np.asarray(observed["oob_mask"], bool)
        graph = graph_audit(48, symmetric_knn(observed_uv_t0, 3), valid)
        p0_all = np.asarray(mf["p0_query"], dtype=np.float32); h_all = np.asarray(mf["h_output_latent"], dtype=np.float32)
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
            "official_teacher_interleaved_12_frame_runs": bool(teacher["official_long_video_protocol"]),
        }
        payload = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "training_approved": False,
            "video_id": manifest["video_id"], "protocol": "official_interleaved23_pj0_anchor_2026-09-28",
            "query_reference_source_frame_index": 14, "output_camera_source_frame_index": 20,
            "offline_teacher_input_source_frames": full_input_indices, "teacher_output_source_frames": teacher_indices,
            "extra_context_not_evaluated_source_frame": 36, "observed_source_frames": observed_indices,
            "evaluated_future_source_frames": list(range(21, 30)),
            "observed_trackcraft_protocol": "nonofficial 7-frame diagnostic; no future-frame padding",
            "gates": gates, "valid_query_count": int(valid.sum()),
            "observed_frame0_anchor_normalized_bbox": quantiles(anchor_obs),
            "teacher_frame0_anchor_normalized_bbox": quantiles(anchor_teacher),
            "cross_context_t0_uv_normalized_frame0_bbox": quantiles(pair_uv),
            "registration_crossfit_normalized_3d_bbox": quantiles(registration),
            "registration_fraction_below_10pct": float((registration <= .10).mean()), "graph": graph,
            "interpretation_limit": "Pj0 anchoring fixes the frame0 output contract but does not validate future pseudo-trajectories or observed/full DA3 consistency",
        }
        np.savez_compressed(
            d / "p0_h_cache_aligned.npz", p0_formal9=p0, h_output_latent_formal9=h_output,
            pseudo_gt_aligned=future.astype(np.float32), pseudo_gt_t0_aligned=aligned[6].astype(np.float32),
            pseudo_gt_validity=np.broadcast_to(valid, (9, 48)), query_uv_t0_source=observed_uv_t0.astype(np.float32),
            registration_crossfit_error_normalized=registration.astype(np.float32),
            coordinate_frame=np.asarray("observed_only_last_camera_source_frame20"),
            pseudo_gt_source=np.asarray("trackcraft3r_official_interleaved23_pj0_anchored"),
        )
    (d / "quality_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
