"""CPU-only native24/source0/full target-surface-point diagnostic preparation.

Uses existing verified MOVi-F shards, never downloads or imports a model.
Preserves the old target identity and old experiment files. MOVi-F has no
WorldTrack sparse annotation set: every source-mask pixel is a candidate material
point; source GT visibility/finite/in-bounds rules are applied, with no erosion
or 48-point subsampling. Other objects/background are outside this object task.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_uniquery4r_movif_gt import (camera_matrices, decode_images, hash_file,
    material_tracks, project, records, verify_shard, write_json)
from prepare_uniquery4r_worldtrack import prepare_one


def full_target_tracks(annot, target_id):
    count = int(annot["metadata/num_frames"][0])
    instances = int(annot["metadata/num_instances"][0])
    h, w = (int(annot[f"metadata/{key}"][0]) for key in ("height", "width"))
    if count != 24 or not 1 <= target_id <= instances:
        raise ValueError("Expected native24 MOVi-F and a valid preserved target")
    segmentation = decode_images(annot["segmentations"])
    local = decode_images(annot["object_coordinates"], channels=3)
    depth = decode_images(annot["depth"])
    if local.dtype != np.uint16 or depth.dtype != np.uint16:
        raise ValueError("Expected simulator uint16 material coordinates/depth")
    mask = segmentation[0] == target_id
    xy = np.argwhere(mask)[:, ::-1]
    if not len(xy):
        raise ValueError("Preserved target not visible in source0; retain as exclusion")
    boxes = annot["instances/bboxes_3d"].reshape(instances, count, 8, 3)[target_id - 1]
    world = material_tracks(local[0, xy[:, 1], xy[:, 0]], boxes)
    k, e = camera_matrices(annot, count, h, w)
    uv, camera = project(world, k, e)
    depth_range = annot["metadata/depth_range"].astype(np.float64)
    distance = depth_range[0] + depth.astype(np.float64) * np.diff(depth_range)[0] / 65535
    visible = np.zeros(camera.shape[:2], bool)
    for t in range(count):
        finite = np.isfinite(uv[t]).all(1) & np.isfinite(camera[t]).all(1)
        raster = np.floor(np.where(np.isfinite(uv[t]), uv[t], -1)).astype(int)
        inside = finite & (camera[t, :, 2] > 0) & (raster[:, 0] >= 0) & (raster[:, 0] < w) & (raster[:, 1] >= 0) & (raster[:, 1] < h)
        ids = np.flatnonzero(inside)
        x, y = raster[ids].T
        d = np.linalg.norm(camera[t, ids], axis=-1)
        visible[t, ids] = (segmentation[t, y, x] == target_id) & (np.abs(d - distance[t, y, x]) < .02 * np.maximum(d, 1e-8))
    anchor = np.linalg.norm(uv[0] - (xy + .5), axis=-1)
    return camera, visible, k, e, mask, xy, {
        "source_mask_candidate_count": len(xy), "source_visible_count": int(visible[0].sum()),
        "source_anchor_p90_pixels": float(np.quantile(anchor, .9)),
        "source_anchor_max_pixels": float(anchor.max()),
        "source_visibility_definition": "simulator same-instance mask, positive depth and 2% ray-depth consistency",
        "erosion": False, "subsampling": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-base", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new directory; preserve all old preparations")
    cfg = json.loads(args.config.read_text())
    old = json.loads((args.existing_base / "logs/gt-preparation-status.json").read_text())
    targets = {r["scene_id"]: r for r in old["scenes"] if r["status"] == "READY"}
    args.output.mkdir(parents=True)
    (args.output / "converted").mkdir()
    rows, sources = [], []
    for entry in cfg["shards"]:
        shard = args.existing_base / "data" / entry["name"]
        if not verify_shard(shard, entry):
            raise ValueError("Publisher checksum failed: " + entry["name"])
        before = hash_file(shard).hex()
        scene_count = 0
        for index, annot in enumerate(records(shard)):
            scene_count += 1
            name = "scene_" + annot["metadata/video_name"][0].decode().replace("/", "_")
            if name not in targets:
                continue
            target = targets[name]
            try:
                camera, visible, k, e, mask, xy, audit = full_target_tracks(annot, target["target_instance_mask_id"])
                source = args.output / "converted" / (name + ".npz")
                np.savez_compressed(source, images_jpeg_bytes=np.asarray(annot["video"]),
                    tracks_XYZ=camera, visibility=visible, fx_fy_cx_cy=k[0, [0, 1, 0, 1], [0, 1, 2, 2]],
                    extrinsics_w2c_gt=e)
                folder = args.output / "inputs/movif_object_diagnostic" / name
                row = prepare_one(source, folder, "movif_object_diagnostic", "movif-simulator", False,
                                  "open_d4rt_endpoint", frame_count=24)
                manifest = json.loads((folder / "manifest.json").read_text())
                manifest.update(unit="simulator scene units; meter equivalence not established",
                    dataset="MOVi-F object diagnostic, not WorldTrack", target_instance_mask_id=target["target_instance_mask_id"],
                    target_asset_id=target["asset_id"], source_publisher_shard=entry["name"],
                    source_publisher_md5_base64=entry["md5_base64"], source_shard_sha256=before,
                    source_record_index=index, full_target_material_audit=audit,
                    point_population="all source-visible target surface pixels; other objects/background excluded")
                write_json(folder / "manifest.json", manifest)
                cv2.imwrite(str(folder / "target_mask_source0.png"), mask.astype(np.uint8) * 255)
                row["material_audit"] = audit
            except Exception as exc:
                row = {"subset": "movif_object_diagnostic", "sequence": name, "status": "FAIL", "error": str(exc)}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("sequence", "status", "query_count", "dynamic_count") if k in row}), flush=True)
        if scene_count != entry["scene_count"] or hash_file(shard).hex() != before:
            raise ValueError("Source count/hash changed")
        sources.append({"shard": entry["name"], "sha256": before, "publisher_md5_passed": True, "source_unchanged": True})
    prepared = [r for r in rows if r["status"] == "CPU_PREPARED"]
    result = {"rows": rows, "sources": sources, "ready_count": len(prepared), "frames": 24,
        "all_attempted_inputs_prepared": len(rows) == len(targets) and len(prepared) == len(targets),
        "gpu_forward_authorized": False, "model_forward_called": False, "query_review_approved": False,
        "worldtrack_benchmark": False}
    write_json(args.output / "preparation.json", result)
    previews = [cv2.resize(cv2.imread(str(args.output / "inputs/movif_object_diagnostic" / r["sequence"] / "query_preview.png")), (384, 384)) for r in prepared]
    if previews:
        tiles = []
        for offset in range(0, len(previews), 4):
            block = previews[offset:offset+4]
            block += [np.zeros_like(block[0])] * (4 - len(block))
            tiles.append(np.hstack(block))
        cv2.imwrite(str(args.output / "query_contact_sheet.jpg"), np.vstack(tiles))
    print(json.dumps({"ready": len(prepared), "attempted": len(rows), "frames": 24, "forward": False}))
    if not result["all_attempted_inputs_prepared"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
