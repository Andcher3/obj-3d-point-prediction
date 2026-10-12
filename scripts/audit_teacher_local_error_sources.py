"""CPU-only diagnostics on retained DAVIS outputs; no inference or data rewriting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_davis_ab_teacher_bundle import mask_hits, crossfit, future_mask_rows
from build_davis_joint_probe_bundle import fit_similarity, apply_similarity, project, transform


VIDEOS = ("car-turn", "drift-turn", "boat", "soccerball")


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"p50": float(np.median(x)), "p90": float(np.quantile(x, .9)),
            "max": float(np.max(x))}


def corrected_fit(source, target):
    x, y = source - source.mean(0), target - target.mean(0)
    u, s, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(np.sum(s * sign) / np.square(x).sum())
    return scale, rotation, target.mean(0) - scale * source.mean(0) @ rotation


def fit_diagnostic(source, target):
    bbox = max(float(np.linalg.norm(np.ptp(target, axis=0))), 1e-12)
    old, _ = crossfit(source, target)
    corrected = np.zeros(len(source))
    reflection_count = 0
    for f in range(4):
        held = np.arange(len(source)) % 4 == f
        x, y = source[~held], target[~held]
        u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
        reflection_count += int(np.linalg.det(u @ vt) < 0)
        corrected[held] = np.linalg.norm(apply_similarity(source[held], corrected_fit(x, y)) - target[held], axis=-1)
    return {"original_crossfit_fraction_bbox": stats(old / bbox),
            "corrected_crossfit_fraction_bbox": stats(corrected / bbox),
            "reflection_folds": reflection_count,
            "difference_max_fraction_bbox": float(np.max(np.abs(corrected - old)) / bbox)}


def camera_from_anchor(xyz, uv_model, scale):
    # Reconstruct effective input K from saved Pj0 and its known integer pixels.
    uv_int = uv_model.astype(np.int64)
    ax = np.column_stack((xyz[:, 0] / xyz[:, 2], np.ones(len(xyz))))
    ay = np.column_stack((xyz[:, 1] / xyz[:, 2], np.ones(len(xyz))))
    fx, cx = np.linalg.lstsq(ax, uv_int[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(ay, uv_int[:, 1], rcond=None)[0]
    k_model = np.asarray([fx, fy, cx, cy])
    return k_model / np.asarray([scale[0], scale[1], scale[0], scale[1]]), float(np.max(np.linalg.norm(project(xyz, k_model) - uv_int, axis=-1)))


def control_checks(mask_root, manifest):
    h, w = manifest["mask_shape_hw"]
    sh, sw = manifest["source_mask_shape_hw"]
    query = np.asarray(manifest["query_sets"]["48"])
    mask = np.asarray(Image.open(mask_root / manifest["video_id"] / "00020.png")) == manifest["png_palette_index"]
    source_query = query * np.asarray([sw / w, sh / h])
    original_mask_indices = np.unique(np.asarray(Image.open(mask_root / manifest["video_id"] / "00020.png")))
    # Execute the actual A/B helper with known points and identity cameras.
    # Its nominal future frame is adjusted to the known t0 image for this control only.
    test_manifest = {**manifest, "t0_source_frame_index": 19}
    xyz = np.column_stack((query, np.ones(len(query))))
    fake_tracks = np.stack((xyz, xyz))
    rows = future_mask_rows(mask_root, test_manifest, fake_tracks,
                            np.tile(np.eye(4), (2, 1, 1)), np.asarray([1, 1, 0, 0]), range(1, 2))
    return {"known_t0_query_count": len(query),
            "processed_hw": [h, w], "source_hw": [sh, sw],
            "actual_ab_helper_hit_count_known_t0": rows[0]["hit_count"],
            "correctly_scaled_hit_count_known_t0": mask_hits(mask, source_query)[0],
            "palette_indices_t0": original_mask_indices.astype(int).tolist(),
            "scaling_roundtrip_max_abs_px": float(np.max(np.abs(source_query / np.asarray([sw/w,sh/h]) - query))),
            "stored_source_query_conversion_max_abs_px": float(np.max(np.abs(source_query - manifest["query_uv_t0_source_image"]))) }


def audit_cached_video(folder, mask_root):
    with np.load(folder / "full22_query_tracks.npz") as t, \
         np.load(folder / "observed7_query_tracks.npz") as o, \
         np.load(folder / "visualization_projection.npz") as p:
        manifest = json.loads((folder / "query_frame0_manifest.json").read_text())
        t_raw = np.asarray(t["trackcraft3r_tracks_raw_runs"], float)
        t_anchor = np.asarray(t["trackcraft3r_tracks_anchored_runs"], float)
        tj = np.asarray(t["frame0_pj_anchor_xyz_runs"], float)
        oj = np.asarray(o["frame0_pj_anchor_xyz"], float)
        delta = np.asarray(t["frame0_anchor_delta_xyz_runs"], float)
        indices = np.asarray(t["interleaved_run_indices"])
        frames = np.asarray(t["source_frame_indices_output"])
        q = np.asarray(t["query_uv_original"], float)
        q_model = np.asarray(t["query_uv_model"], float)
        ph, pw = (int(x) for x in p["processed_resolution_hw"])
        mh, mw = (int(x) for x in t["model_resolution_hw"])
        k, k_residual = camera_from_anchor(tj[0], q_model, np.asarray([mw/pw, mh/ph]))
        merged_raw = np.asarray(t["trackcraft3r_tracks_raw_interleaved"], float)
        merged_anchored = np.asarray(t["trackcraft3r_tracks"], float)
        sum_raw = np.zeros((23, 48, 3)); sum_anc = np.zeros_like(sum_raw)
        count = np.zeros(23)
        for r, idx in enumerate(indices):
            sum_raw[idx] += t_raw[r]; sum_anc[idx] += t_anchor[r]; count[idx] += 1
        if not np.all(count > 0):
            raise ValueError("interleaving leaves uncovered frames")
        reconstructed_raw = sum_raw / count[:, None, None]
        reconstructed_anc = sum_anc / count[:, None, None]
        changed_steps = np.diff(merged_anchored - merged_raw, axis=0)
        distances = np.linalg.norm(oj[:, None] - oj[None], axis=-1)
        median_distance = float(np.median(distances[np.triu_indices(48, 1)]))
        uv = np.asarray(p["teacher_uv_processed"], float)
        uv_valid = np.asarray(p["teacher_valid"], bool)
        hit_rows = []
        for idx, frame in enumerate(frames):
            m = np.asarray(Image.open(mask_root / folder.name / f"{int(frame):05d}.png")) == 1
            source_scale = np.asarray([m.shape[1]/pw, m.shape[0]/ph])
            # mask_hits does not filter invalid projections; this wrapper does.
            v = uv_valid[idx] & np.isfinite(uv[idx]).all(1)
            correct = mask_hits(m, uv[idx][v] * source_scale)[0]
            unscaled = mask_hits(m, uv[idx][v])[0]
            hit_rows.append({"source_frame": int(frame), "correct_count_of_48": correct,
                             "unscaled_count_of_48": unscaled})
        raw0_errors = [stats(np.linalg.norm(project(run[0], k) - q, axis=-1)) for run in t_raw]
        anc0_errors = [stats(np.linalg.norm(project(run[0], k) - q, axis=-1)) for run in t_anchor]
        # Pj0 is the saved input pointmap sample, before decoded tracking output.
        geometry = fit_diagnostic(tj[0], oj)
        pair_o = distances[np.triu_indices(48, 1)]
        pair_t = np.linalg.norm(tj[0][:, None] - tj[0][None], axis=-1)[np.triu_indices(48, 1)]
        pair_scale = float(np.median(pair_o / np.maximum(pair_t, 1e-12)))
        geometry["pairwise_distance_relative_difference_after_median_scale"] = stats(np.abs(pair_t * pair_scale - pair_o) / np.maximum(pair_o, 1e-12))
        geometry["interpretation"] = "Same source14 query pixels; input Pj0 geometry discrepancy, not 3D ground-truth error. Context/model-grid intrinsics sampling remain coupled."
        return {"source_protocol": "2026-09-28 frame0-query official interleaving; not 2026-09-29 Route A",
                "arrays_finite": bool(np.isfinite(t_raw).all() and np.isfinite(t_anchor).all() and np.isfinite(oj).all()),
                "query_same_processed_pixels_obs_teacher_max_abs": float(np.max(np.abs(q - o["query_uv_original"]))),
                "inferred_effective_K_from_saved_Pj0": k.tolist(),
                "inferred_K_fit_max_px": k_residual,
                "raw_teacher_frame0_query_error_processed_px": raw0_errors,
                "anchored_teacher_frame0_query_error_processed_px": anc0_errors,
                "input_geometry_same_frame14": geometry,
                "interleave_index_exact_match": bool(np.array_equal(indices, np.asarray([list(range(0,23,2)),[0]+list(range(1,23,2))]))),
                "source_frames_exact_match": bool(np.array_equal(frames, np.arange(14,36))),
                "raw_merge_reproduction_max_abs": float(np.max(np.abs(reconstructed_raw[:22] - merged_raw))),
                "anchored_merge_reproduction_max_abs": float(np.max(np.abs(reconstructed_anc[:22] - merged_anchored))),
                "within_run_constant_offset_residual_max_abs": float(np.max(np.abs(t_anchor - t_raw - delta[:,None]))),
                "teacher_start_exactly_input_Pj0_max_abs": float(np.max(np.abs(t_anchor[:,0] - tj))),
                "different_anchor_offsets_between_runs_norm": stats(np.linalg.norm(delta[0]-delta[1],axis=-1)),
                "anchor_induced_adjacent_step_change_norm": stats(np.linalg.norm(changed_steps,axis=-1)),
                "anchor_induced_adjacent_step_p90_fraction_median_query_spacing": float(np.quantile(np.linalg.norm(changed_steps,axis=-1),.9) / median_distance),
                "mask_recomputation": hit_rows}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a=p.parse_args()
    results = a.project_root / "experiences_results"
    ab = results / "surface-relation-davis-ab-teacher-2026-09-29/files"
    old = results / "surface-relation-davis-official-interleaved-pj0-2026-09-28"
    mask_root = results / "surface-relation-candidate-videos-2026-09-27/selected-frames/DAVIS/Annotations/480p"
    out={"status":"LOCAL_CACHE_AUDIT_NO_GPU_NO_TRAINING", "videos":{}, "analytic_checks":{}}
    rng=np.random.default_rng(101); x=rng.normal(size=(48,3))
    r=np.asarray([[0,-1,0],[1,0,0],[0,0,1]],float)
    y=2.3*x@r + np.asarray([2.,3.,4.])
    out["analytic_checks"]["similarity_known_transform_max_abs"] = float(np.max(np.abs(apply_similarity(x,fit_similarity(x,y))-y)))
    y_reflected=x*np.asarray([-1,1,1])
    old_loss=float(np.mean(np.square(apply_similarity(x,fit_similarity(x,y_reflected))-y_reflected)))
    correct_loss=float(np.mean(np.square(apply_similarity(x,corrected_fit(x,y_reflected))-y_reflected)))
    out["analytic_checks"]["reflection_scale_formula"]={"original_squared_error":old_loss,"corrected_squared_error":correct_loss,"scale_bug_confirmed":old_loss>correct_loss+1e-12}
    # Use the actual project coordinate helpers for a positive-depth roundtrip.
    points=np.column_stack((rng.normal(size=(48,2)),rng.uniform(2,4,size=48)))
    pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=[.1,.2,.3]
    out["analytic_checks"]["camera_roundtrip_max_abs"] = float(np.max(np.abs(transform(np.linalg.inv(pose),transform(pose,points))-points)))
    source_files=[]
    for v in VIDEOS:
        manifest_path=ab/"common"/v/"query_t0_manifest.json"
        m=json.loads(manifest_path.read_text())
        row={"route_a_t0_mask_interface_control":control_checks(mask_root,m),
             "cached_frame0_teacher_diagnostics":audit_cached_video(old/v,mask_root)}
        mapping=json.loads((ab/"common"/v/"query_frame0_mapping.json").read_text())
        mapped=np.asarray(mapping["query_sets"]["48"],float)
        row["route_a_observed_query_mapping_metadata_only"]={
            "unique_frame0_queries":int(len(np.unique(mapped,axis=0))),
            "mapping_p90_fraction_bbox":mapping["match_p90_normalized_bbox"],
            "mapping_max_fraction_bbox":mapping["match_max_normalized_bbox"],
            "point_identity_independently_verified":False}
        new_d=json.loads((ab/"common"/v/"official_interleaved_derivation.json").read_text())
        row["route_a_metadata_only"]={"input_frames":new_d["source_frame_indices_input"],
            "output_frames":new_d["source_frame_indices_output"],
            "same_input_Pj0_between_runs_max_abs":new_d["frame0_pj_between_runs_max_abs"],
            "original_merge_displacement_change_max_abs":new_d["temporal_displacement_max_abs_change_after_merged_anchor"],
            "raw_npz_available_in_route_a_local_directory":bool(list((ab/"common"/v).glob("*.npz")))}
        old_d=json.loads((old/v/"future_projection_diagnostic.json").read_text())
        row["previous_pose_diagnostic_only"]={"pointmap_roundtrip_max_px":max(old_d["predictor_pointmap_coordinate_roundtrip"]["roundtrip_max_error_px_per_frame"]),
            "loader_intrinsics_difference_max_abs":old_d["load_npz_data_intrinsics_max_abs_difference_from_stored"],
            "loader_pose_difference_max_abs":old_d["load_npz_data_w2c_max_abs_difference_from_stored"],
            "frame0_pose_identity_difference_max_abs":old_d["stored_frame0_w2c_max_abs_difference_identity"]}
        row["previous_future_projection_alternatives_metadata_only"]={
            mode:old_d["official_merged_anchored"][mode]["mask_hits_of_48"][7:16]
            for mode in ("w2c_per_frame","identity","inverse_as_w2c","w2c0_fixed")}
        out["videos"][v]=row
        source_files.extend([manifest_path,old/v/"full22_query_tracks.npz",old/v/"observed7_query_tracks.npz",old/v/"visualization_projection.npz"])
    out["inputs_sha256"]={str(p.relative_to(a.project_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    out["limitations"]=["Route A/B raw NPZ and DA3 user arrays are not present locally; their true future mask counts cannot be recomputed here.",
        "No manual surface-point identity labels or independent true geometry are used; discrepancy is not automatically error.",
        "Known-coordinate controls and retained frame0 caches are different evidence sources; no result is substituted for Route A quality gates."]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for v,row in out["videos"].items():
        control=row["route_a_t0_mask_interface_control"];d=row["cached_frame0_teacher_diagnostics"]
        print(json.dumps({"video":v,"known_query_mask_hit_buggy_correct":[control["actual_ab_helper_hit_count_known_t0"],control["correctly_scaled_hit_count_known_t0"]],
            "input_geometry_crossfit_p90":d["input_geometry_same_frame14"]["corrected_crossfit_fraction_bbox"]["p90"],
            "anchor_step_change_p90":d["anchor_induced_adjacent_step_change_norm"]["p90"],
            "future21_29_correct_mask_counts":[r["correct_count_of_48"] for r in d["mask_recomputation"] if 21<=r["source_frame"]<=29]}))
    print(json.dumps(out["analytic_checks"]))


if __name__ == "__main__":main()
