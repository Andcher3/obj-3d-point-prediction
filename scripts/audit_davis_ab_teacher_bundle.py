#!/usr/bin/env python3
"""Independent quality gates for DAVIS t0-first and full-context teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from build_davis_joint_probe_bundle import (
    apply_similarity, fit_similarity, graph_audit, project, quantiles, symmetric_knn, transform,
)


def relative_w2c(w2c: np.ndarray) -> np.ndarray:
    return np.einsum("tij,jk->tik", w2c, np.linalg.inv(w2c[0]))


def load_mask(root: Path, video: str, frame: int, instance: int) -> np.ndarray:
    array = np.asarray(Image.open(root / video / f"{frame:05d}.png"))
    if array.ndim != 2:
        raise ValueError("DAVIS identity audit requires palette/index masks")
    return array == instance


def mask_hits(mask: np.ndarray, uv: np.ndarray) -> tuple[int, float]:
    xy = np.rint(uv).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    hit = np.zeros(len(xy), dtype=bool)
    hit[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return int(hit.sum()), float(hit.mean())


def crossfit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    error = np.full(len(source), np.nan, dtype=np.float64)
    rows = []
    for fold in range(4):
        held = np.arange(len(source)) % 4 == fold
        fit = fit_similarity(source[~held], target[~held])
        error[held] = np.linalg.norm(apply_similarity(source[held], fit) - target[held], axis=-1)
        rows.append({"fold": fold, "held_error_m": quantiles(error[held])})
    return error, rows


def future_mask_rows(mask_root: Path, manifest: dict, tracks: np.ndarray, w2c: np.ndarray, k: np.ndarray, slots: range) -> list[dict]:
    matrices = relative_w2c(w2c)
    rows = []
    for slot in slots:
        uv = project(transform(matrices[slot], tracks[slot]), k)
        frame = int(np.asarray(manifest["t0_source_frame_index"])) + (slot - slots.start + 1)
        count, fraction = mask_hits(load_mask(mask_root, manifest["video_id"], frame, int(manifest["png_palette_index"])), uv)
        rows.append({"source_frame_index": frame, "hit_count": count, "fraction": fraction})
    return rows


def link_inputs(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("observed7_user.npz", "observed7_query_tracks.npz"):
        link = target / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to((source / name).resolve())


def save_cache(directory: Path, p0: np.ndarray, h: np.ndarray, gt: np.ndarray, t0: np.ndarray,
               valid: np.ndarray, anchor_px: np.ndarray, query: np.ndarray, observed_t0: np.ndarray,
               registration: np.ndarray, source: str) -> None:
    np.savez_compressed(
        directory / "p0_h_cache_aligned.npz",
        p0_formal9=p0.astype(np.float32), h_output_latent_formal9=h.astype(np.float32),
        pseudo_gt_aligned=gt.astype(np.float32), pseudo_gt_t0_aligned=t0.astype(np.float32),
        pseudo_gt_validity=np.broadcast_to(valid, (9, 48)),
        query_anchor_match_error_px=anchor_px.astype(np.float32),
        query_uv_t0_source=query.astype(np.float32), observed_only_trackcraft_t0=observed_t0.astype(np.float32),
        registration_crossfit_error_normalized=registration.astype(np.float32),
        coordinate_frame=np.asarray("observed_only_last_camera_source_frame20"),
        prediction_source=np.asarray("motionforesight_observed_only_7_frames"),
        pseudo_gt_source=np.asarray(source), future_frames_supplied_to_model=np.asarray(False),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--route-a-output", type=Path, required=True)
    ap.add_argument("--route-b-output", type=Path, required=True)
    args = ap.parse_args(); d = args.video_dir
    manifest = json.loads((d / "query_t0_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((d / "query_frame0_mapping.json").read_text(encoding="utf-8"))
    mf_meta = json.loads((d / "motionforesight_observed_only.json").read_text(encoding="utf-8"))
    query_t0 = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
    query_frame0 = np.asarray(mapping["query_sets"]["48"], dtype=np.float64)
    bbox2 = max(float(manifest["bbox_diagonal_px"]), 1e-8)

    with np.load(d / "observed7_user.npz", allow_pickle=True) as obs, \
         np.load(d / "observed7_query_tracks.npz") as observed, \
         np.load(d / "motionforesight_observed_only_arrays.npz") as mf, \
         np.load(d / "route_a_full23_user.npz", allow_pickle=True) as a_user, \
         np.load(d / "route_a_full22_query_tracks.npz") as a_track, \
         np.load(d / "route_b_full23_user.npz", allow_pickle=True) as b_user, \
         np.load(d / "route_b_full22_query_tracks.npz") as b_track:
        if not np.array_equal(obs["frame_indices"], np.arange(14, 21)):
            raise ValueError("observed frames must be source 14..20")
        if not np.array_equal(a_user["frame_indices"], np.arange(20, 43)):
            raise ValueError("route A must use source 20..42")
        if not np.array_equal(b_user["frame_indices"], np.arange(14, 37)):
            raise ValueError("route B must use source 14..36")
        if not np.allclose(observed["query_uv_original"], query_frame0, atol=1e-4):
            raise ValueError("observed query order mismatch")
        if not np.allclose(a_track["query_uv_original"], query_t0, atol=1e-4):
            raise ValueError("route A query order mismatch")
        if not np.allclose(mf["query_uv_original"], query_frame0, atol=1e-4):
            raise ValueError("MotionForesight query order mismatch")

        obs_world = np.asarray(observed["trackcraft3r_tracks"], dtype=np.float64)
        observed_t0 = transform(np.asarray(obs["extrinsics_w2c"], dtype=np.float64)[-1], obs_world)[-1]
        a_tracks = np.asarray(a_track["trackcraft3r_tracks"], dtype=np.float64)
        a_t0 = a_tracks[0]
        a_k = np.asarray(a_user["fx_fy_cx_cy"], dtype=np.float64)
        a_anchor_px = np.linalg.norm(project(a_t0, a_k) - query_t0, axis=-1)
        registration_m, folds = crossfit(a_t0, observed_t0)
        bbox3 = max(float(np.linalg.norm(np.ptp(observed_t0, axis=0))), 1e-8)
        registration = registration_m / bbox3
        common_fit = fit_similarity(a_t0, observed_t0)
        common_t0 = apply_similarity(a_t0, common_fit)
        a_gt = apply_similarity(a_tracks[1:10], common_fit)

        b_raw = np.asarray(b_track["trackcraft3r_tracks"], dtype=np.float64)
        b_w2c = np.asarray(b_user["extrinsics_w2c"], dtype=np.float64)[:22]
        b_cam20 = transform(relative_w2c(b_w2c)[6], b_raw)
        b_k = np.asarray(b_user["fx_fy_cx_cy"], dtype=np.float64)
        b_uv_t0 = project(b_cam20[6], b_k)
        cost = np.linalg.norm(b_uv_t0[:, None] - query_t0[None], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        order = np.empty(48, dtype=np.int64); order[cols] = rows
        match_px = cost[order, np.arange(48)]
        b_cam20 = b_cam20[:, order]
        b_raw = b_raw[:, order]
        b_common = a_t0[None] + (b_cam20 - b_cam20[6:7])
        b_gt = apply_similarity(b_common[7:16], common_fit)

        observed_valid = np.isfinite(obs_world).all(-1).all(0)
        a_valid = observed_valid & np.isfinite(a_tracks[:10]).all(-1).all(0) & (a_tracks[:10, :, 2] > 0).all(0)
        b_valid = observed_valid & np.isfinite(b_cam20[6:16]).all(-1).all(0) & (b_cam20[6:16, :, 2] > 0).all(0)
        if "oob_mask" in a_track: a_valid &= np.asarray(a_track["oob_mask"], bool)
        if "oob_mask" in b_track: b_valid &= np.asarray(b_track["oob_mask"], bool)[order]
        graph_a = graph_audit(48, symmetric_knn(query_t0, 3), a_valid)
        graph_b = graph_audit(48, symmetric_knn(query_t0, 3), b_valid)
        p0_all = np.asarray(mf["p0_query"], dtype=np.float64); h_all = np.asarray(mf["h_output_latent"], dtype=np.float64)
        p0, h = p0_all[7:16], h_all[7:16]
        no_future = not bool(mf_meta.get("input", {}).get("future_frame_tensor_supplied", True))
        shared = {
            "future_frames_not_supplied_to_model": no_future,
            "common_t0_registration_p90_le_10pct": float(np.quantile(registration, .9)) <= .10,
            "common_t0_registration_90pct_below_10pct": float((registration <= .10).mean()) >= .90,
            "common_t0_registration_max_le_20pct": float(registration.max()) <= .20,
            "finite_p0_h": bool(np.isfinite(p0).all() and np.isfinite(h).all()),
        }
        gates_a = {**shared,
            "direct_t0_query_order": True,
            "t0_anchor_p90_le_5pct": float(np.quantile(a_anchor_px / bbox2, .9)) <= .05,
            "t0_anchor_max_le_10pct": float((a_anchor_px / bbox2).max()) <= .10,
            "valid_queries_at_least_44": int(a_valid.sum()) >= 44,
            "valid_graph_connected": bool(graph_a["connected"]),
            "finite_teacher": bool(np.isfinite(a_gt).all()),
        }
        gates_b = {**shared,
            "hungarian_one_to_one_48": len(np.unique(order)) == 48,
            "identity_match_p90_le_5pct": float(np.quantile(match_px / bbox2, .9)) <= .05,
            "identity_match_max_le_10pct": float((match_px / bbox2).max()) <= .10,
            "valid_queries_at_least_44": int(b_valid.sum()) >= 44,
            "valid_graph_connected": bool(graph_b["connected"]),
            "finite_teacher": bool(np.isfinite(b_gt).all()),
        }
        status_a = "PASS" if all(gates_a.values()) else "FAIL"
        status_b = "PASS" if all(gates_b.values()) else "FAIL"
        a_masks = future_mask_rows(args.mask_root, manifest, a_tracks, np.asarray(a_user["extrinsics_w2c"], dtype=np.float64), a_k, range(1, 10))
        b_masks = future_mask_rows(args.mask_root, manifest, b_raw, b_w2c, b_k, range(7, 16))
        common = {
            "video_id": manifest["video_id"], "preview_mp4_fps_used": False,
            "time_basis": "consecutive DAVIS source frame indices; capture fps unavailable",
            "common_t0_geometry_source": "route A DA3 source-frame-20 point geometry",
            "common_t0_registration_normalized_3d_bbox": quantiles(registration),
            "common_t0_registration_fraction_below_10pct": float((registration <= .10).mean()),
            "registration_folds": folds,
        }
        audit_a = {**common, "route": "A_t0_first", "status": status_a, "training_approved": status_a == "PASS",
            "gates": gates_a, "valid_query_count": int(a_valid.sum()),
            "t0_anchor_px": quantiles(a_anchor_px), "t0_anchor_normalized_bbox": quantiles(a_anchor_px / bbox2),
            "graph": graph_a, "future_mask_hits": a_masks,
            "teacher_input_frames": list(range(20, 43)), "teacher_future_frames": list(range(21, 30))}
        audit_b = {**common, "route": "B_full_context_matched", "status": status_b, "training_approved": status_b == "PASS",
            "gates": gates_b, "valid_query_count": int(b_valid.sum()),
            "identity_match_px": quantiles(match_px), "identity_match_normalized_bbox": quantiles(match_px / bbox2),
            "graph": graph_b, "future_mask_hits": b_masks,
            "teacher_input_frames": list(range(14, 37)), "teacher_future_frames": list(range(21, 30)),
            "edge_target": "(X_F(t)-X_F(t0)) edge change plus common route-A t0 edge"}

        args.route_a_output.mkdir(parents=True, exist_ok=True); args.route_b_output.mkdir(parents=True, exist_ok=True)
        (args.route_a_output / "quality_audit.json").write_text(json.dumps(audit_a, indent=2) + "\n")
        (args.route_b_output / "quality_audit.json").write_text(json.dumps(audit_b, indent=2) + "\n")
        if status_a == "PASS":
            link_inputs(d, args.route_a_output)
            save_cache(args.route_a_output, p0, h, a_gt, common_t0, a_valid, a_anchor_px, query_t0, observed_t0, registration, "route_A_t0_first_trackcraft3r")
        if status_b == "PASS":
            link_inputs(d, args.route_b_output)
            save_cache(args.route_b_output, p0, h, b_gt, common_t0, b_valid, match_px, query_t0, observed_t0, registration, "route_B_full_context_matched_edge_change")
    print(json.dumps({"video_id": manifest["video_id"], "route_a": status_a, "route_b": status_b}, indent=2))


if __name__ == "__main__":
    main()
