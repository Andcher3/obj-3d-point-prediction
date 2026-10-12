"""CPU-only density sampling, retaining original target, query6 and 22 frames.

Nominal whole-image budget ~1000, fixed-phase square lattice inside t0 mask.
Below 48 valid lattice points, reuse the exact original48 query and GT files.
No downloads, Torch imports, model load, future-based selection or GPU use.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from prepare_uniquery4r_movif_gt import (camera_matrices, decode_images, hash_file,
    material_tracks, project, records, verify_shard, write_input_hashes, write_json)
from uniquery4r_worldtrack_protocol import dynamic_mask
from run_uniquery4r_movif_experiment1 import input_audit


def grid_pixels(mask, whole_image_budget=1000):
    """Fixed image-origin lattice; density is never fitted to the target shape."""
    h, w = mask.shape
    if whole_image_budget <= 0:
        raise ValueError("Positive nominal whole-image budget required")
    step = float(np.sqrt(h * w / whole_image_budget))
    # Grid centers lie at half a cell. Select the pixel containing each center.
    xs = np.floor(np.arange(step / 2, w, step)).astype(int)
    ys = np.floor(np.arange(step / 2, h, step)).astype(int)
    if len(np.unique(xs)) != len(xs) or len(np.unique(ys)) != len(ys):
        raise ValueError("Spacing too small for distinct raster pixels")
    x, y = np.meshgrid(xs, ys)
    xy = np.c_[x.ravel(), y.ravel()]
    return xy[mask[xy[:, 1], xy[:, 0]].astype(bool)], step, len(xy)


def use_original48(valid_count):
    return valid_count < 48


def selected_tracks(annot, target_id, xy, count=22, query_frame=6):
    total = int(annot["metadata/num_frames"][0])
    instances = int(annot["metadata/num_instances"][0])
    h, w = (int(annot[f"metadata/{key}"][0]) for key in ("height", "width"))
    if total != 24 or not 1 <= target_id <= instances or count != 22 or query_frame != 6:
        raise ValueError("Expected MOVi-F native24, original22/query6 and preserved target")
    segmentation = decode_images(annot["segmentations"])
    local = decode_images(annot["object_coordinates"], channels=3)
    encoded_depth = decode_images(annot["depth"])
    if local.dtype != np.uint16 or encoded_depth.dtype != np.uint16:
        raise ValueError("Expected simulator uint16 point/depth data")
    boxes = annot["instances/bboxes_3d"].reshape(instances, total, 8, 3)[target_id-1]
    object_points = local[query_frame, xy[:, 1], xy[:, 0]]
    world = material_tracks(object_points, boxes)[:count]
    k, e = camera_matrices(annot, total, h, w)
    k, e = k[:count], e[:count]
    uv, camera = project(world, k, e)
    limits = annot["metadata/depth_range"].astype(np.float64)
    distance = limits[0] + encoded_depth.astype(np.float64) * np.diff(limits)[0] / 65535
    visible = np.zeros((count, len(xy)), bool)
    for t in range(count):
        raster = np.floor(np.where(np.isfinite(uv[t]), uv[t], -1)).astype(int)
        inside = (np.isfinite(camera[t]).all(1) & (camera[t, :, 2] > 0)
            & (raster[:, 0] >= 0) & (raster[:, 0] < w) & (raster[:, 1] >= 0) & (raster[:, 1] < h))
        ids = np.flatnonzero(inside)
        x, y = raster[ids].T
        d = np.linalg.norm(camera[t, ids], axis=-1)
        visible[t, ids] = (segmentation[t, y, x] == target_id) & (np.abs(d-distance[t, y, x]) < .02 * np.maximum(d, 1e-8))
    first = np.einsum("ij,tqj->tqi", e[0, :3, :3], world) + e[0, :3, 3]
    last = np.einsum("ij,tqj->tqi", e[query_frame, :3, :3], world) + e[query_frame, :3, 3]
    depth_relative = np.abs(np.linalg.norm(camera[query_frame], axis=1) - distance[query_frame, xy[:, 1], xy[:, 0]]) / np.maximum(distance[query_frame, xy[:, 1], xy[:, 0]], 1e-8)
    return dict(tracks_world=world, tracks_first_camera=first, tracks_last_observed_camera=last,
        tracks_2d=uv, visibility=visible, validity=np.ones_like(visible), intrinsic_gt=k,
        extrinsic_w2c_gt=e, object_coordinates_t0_uint16=object_points, target_bboxes_world=boxes[:count],
        target_instance_mask_id=np.int32(target_id), source_frame_indices=np.arange(count), timestamps_s=np.arange(count)/12), depth_relative


def prepare_scene(existing, output, cfg, annot, old):
    name = old["scene_id"]
    prior = existing / "inputs" / name
    folder = output / "inputs" / name
    old_manifest = json.loads((prior / "queries.json").read_text())
    if old_manifest["query_model_frame_index"] != 6 or old_manifest["source_frame_indices"] != list(range(22)):
        raise ValueError("Original experiment protocol changed")
    target = old["target_instance_mask_id"]
    mask = decode_images(annot["segmentations"])[6] == target
    xy, step, full_count = grid_pixels(mask, cfg["whole_image_nominal_query_count"])
    if not len(xy):
        raise ValueError("No grid candidates on target at query6")
    pack, depth_relative = selected_tracks(annot, target, xy)
    uv = pack["tracks_2d"][6]
    ph, pw = old_manifest["processed_resolution_hw"]
    h, w = mask.shape
    processed = (uv+.5) * [pw/w, ph/h] - .5
    norm = processed / [pw-1, ph-1]
    # Selection uses only t0 information. Do not filter future visibility.
    accepted = (pack["visibility"][6] & np.isfinite(uv).all(1)
        & ((norm >= 0) & (norm <= 1)).all(1)
        & (np.linalg.norm(uv - (xy+.5), axis=1) <= cfg["cpu_audit"]["gt_t0_projection_max_pixels"]))
    valid_count = int(accepted.sum())
    fallback = use_original48(valid_count)
    folder.mkdir(parents=True)
    shutil.copytree(prior / "rgb", folder / "rgb")
    shutil.copy2(prior / "target_mask_t0.png", folder / "target_mask_t0.png")
    old_query_hash = hash_file(prior / "queries.json").hex()
    old_gt_hash = hash_file(prior / "gt_tracks.npz").hex()
    if fallback:
        shutil.copy2(prior / "queries.json", folder / "queries.json")
        shutil.copy2(prior / "gt_tracks.npz", folder / "gt_tracks.npz")
        with np.load(folder / "gt_tracks.npz", allow_pickle=False) as f:
            pack = {key: f[key] for key in f.files}
        manifest = old_manifest
        assert hash_file(folder / "queries.json").hex() == old_query_hash
        assert hash_file(folder / "gt_tracks.npz").hex() == old_gt_hash
    else:
        for key in ("tracks_world", "tracks_first_camera", "tracks_last_observed_camera", "tracks_2d", "visibility", "validity"):
            pack[key] = pack[key][:, accepted]
        pack["object_coordinates_t0_uint16"] = pack["object_coordinates_t0_uint16"][accepted]
        np.savez_compressed(folder / "gt_tracks.npz", **pack)
        manifest = dict(old_manifest)
        manifest.update(query_uv_source_pixels=uv[accepted].tolist(), query_uv_normalized_01=norm[accepted].tolist(),
            query_pixel_sample_xy=xy[accepted].tolist(), query_user_review="pending new preview approval; GPU disabled")
        write_json(folder / "queries.json", manifest)
    count = len(manifest["query_pixel_sample_xy"])
    query_uv = np.asarray(manifest["query_uv_source_pixels"])
    sample_xy = np.asarray(manifest["query_pixel_sample_xy"])
    anchor_error = np.linalg.norm(pack["tracks_2d"][6] - (sample_xy+.5), axis=1)
    if not np.isfinite(pack["tracks_first_camera"]).all() or not pack["visibility"][6].all():
        raise ValueError("Selected GT is invalid or query not visible at t0")
    if anchor_error.max() > cfg["cpu_audit"]["gt_t0_projection_max_pixels"]:
        raise ValueError("GT source projection failed")
    if not fallback and np.quantile(depth_relative[accepted], .9) > cfg["cpu_audit"]["gt_t0_depth_relative_p90"]:
        raise ValueError("GT source depth failed")
    summary = {**old, "status": "READY", "query_count": count, "query_frame": 6, "frames": 22,
        "sampling_mode": "original48_exact_reuse" if fallback else "fixed_image_lattice",
        "whole_image_nominal_count": cfg["whole_image_nominal_query_count"], "whole_image_lattice_actual_count": full_count,
        "spacing_source_pixels": step, "spacing_processed_pixels": step*pw/w,
        "query6_target_mask_pixels": int(mask.sum()), "grid_mask_candidate_count": len(xy), "grid_t0_valid_count": valid_count,
        "grid_t0_excluded_count": len(xy)-valid_count, "original_query_sha256": old_query_hash, "original_gt_sha256": old_gt_hash,
        "fallback_byte_exact": fallback, "all_query_visible_at_t0": True,
        "gt_t0_pixel_anchor_error_max": float(anchor_error.max()), "gt_t0_pixel_anchor_error_p90": float(np.quantile(anchor_error, .9)),
        "gt_t0_depth_relative_p90": old["gt_t0_depth_relative_p90"] if fallback else float(np.quantile(depth_relative[accepted], .9)),
        "future_visible_fraction": float(pack["visibility"][7:].mean()),
        "gt_tracks_shape": list(pack["tracks_first_camera"].shape), "dynamic_query_count": int(dynamic_mask(pack["tracks_first_camera"]).sum()),
        "future_selection_or_visibility_filter": False, "model_forward_called": False, "query_review_approved": False}
    write_json(folder / "sampling.json", summary)
    write_json(folder / "gt_audit.json", summary)
    image = cv2.imread(str(folder / "rgb" / "006.png"))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_NEAREST)
    for i, (u, v) in enumerate(query_uv):
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[i*179//count, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(image, (round((u+.5)*3-.5), round((v+.5)*3-.5)), 3, color, -1)
    image = cv2.copyMakeBorder(image, 55, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, f"{name} | query6 | N={count} | {'old48' if fallback else 'grid~1000/image'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
    cv2.imwrite(str(folder / "query_preview.png"), image)
    write_input_hashes(output, folder)
    if hash_file(prior / "queries.json").hex() != old_query_hash or hash_file(prior / "gt_tracks.npz").hex() != old_gt_hash:
        raise ValueError("Original input changed during preparation")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing-base", required=True, type=Path)
    ap.add_argument("--original-config", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing preparations; choose a new output directory")
    cfg = json.loads(args.config.read_text())
    original_cfg = json.loads(args.original_config.read_text())
    old = json.loads((args.existing_base / "logs/gt-preparation-status.json").read_text())
    targets = {r["scene_id"]: r for r in old["scenes"] if r["status"] == "READY"}
    (args.output / "logs").mkdir(parents=True)
    rows, source_audits = [], []
    for entry in original_cfg["shards"]:
        path = args.existing_base / "data" / entry["name"]
        if not verify_shard(path, entry):
            raise ValueError("Publisher source checksum failed: " + entry["name"])
        before = hash_file(path).hex()
        total = 0
        for annot in records(path):
            total += 1
            name = "scene_" + annot["metadata/video_name"][0].decode().replace("/", "_")
            if name not in targets:
                continue
            result = prepare_scene(args.existing_base, args.output, cfg, annot, targets[name])
            rows.append(result)
            print(json.dumps({k: result[k] for k in ("scene_id", "query_count", "grid_t0_valid_count", "sampling_mode")}), flush=True)
            write_json(args.output / "logs/gt-preparation-status.json", {"complete": False, "scenes": rows})
        if total != entry["scene_count"] or hash_file(path).hex() != before:
            raise ValueError("Source count/hash changed")
        source_audits.append({"shard": entry["name"], "publisher_md5_passed": True, "source_sha256": before, "source_unchanged": True})
    if len(rows) != len(targets):
        raise ValueError("Missing original target scenes")
    status = {"complete": True, "scenes": rows, "scenes_ready": len(rows), "gt_coordinate_checks_passed": True,
        "frames": 22, "query_frame": 6, "sources": source_audits, "query_review_approved": False,
        "gpu_forward_authorized": False, "model_forward_called": False, "worldtrack_exact_reproduction": False,
        "unit": "simulator scene units; meter equivalence not established"}
    write_json(args.output / "logs/gt-preparation-status.json", status)
    write_json(args.output / "preparation.json", status)
    input_audit(args.output)
    tiles = [cv2.resize(cv2.imread(str(args.output / "inputs" / r["scene_id"] / "query_preview.png")), (384, 412)) for r in rows]
    bands = []
    for i in range(0, len(tiles), 4):
        block = tiles[i:i+4]
        block += [np.zeros_like(block[0])] * (4-len(block))
        bands.append(np.hstack(block))
    cv2.imwrite(str(args.output / "query_contact_sheet.jpg"), np.vstack(bands))
    print(json.dumps({"prepared": len(rows), "fallback48": sum(r["fallback_byte_exact"] for r in rows), "model_forward_called": False}))


if __name__ == "__main__":
    main()
