"""Small official MOVi-F subset -> object queries and simulator GT, CPU only.

No Torch import or model forward. Read TF Example wire format without installing
TensorFlow. Verify entire source shards against the publisher MD5 before parsing.
Material point construction follows Kubric challenges/point_tracking/dataset.py.
"""
import argparse
import base64
import hashlib
import itertools
import json
import os
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.digest()


def write_input_hashes(base, folder):
    # Never hash the manifest itself: repeat preparation must remain auditable.
    hashes = {str(p.relative_to(base)): hash_file(p).hex() for p in sorted(folder.rglob("*"))
              if p.is_file() and p.name != "sha256.json"}
    write_json(folder / "sha256.json", hashes)
    return hashes


def verify_shard(path, entry):
    if not path.exists() or path.stat().st_size != entry["bytes"] or Path(str(path) + ".aria2").exists():
        return False
    return base64.b64encode(hash_file(path, "md5")).decode() == entry["md5_base64"]


def download(base, cfg, indices):
    data = base / "data"
    data.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in indices:
        entry = cfg["shards"][index]
        path = data / entry["name"]
        if verify_shard(path, entry):
            print("SOURCE_ALREADY_VERIFIED", entry["name"], flush=True)
            continue
        entries.append(cfg["source_base"] + "/" + entry["name"] + "?generation=" + entry["generation"]
                       + "\n  dir=" + str(data) + "\n  out=" + entry["name"]
                       + "\n  checksum=md5=" + base64.b64decode(entry["md5_base64"]).hex() + "\n")
    if entries:
        selection_tag = "-".join(map(str, indices))
        queue = base / ("download-queue-" + selection_tag + ".txt")
        queue.write_text("".join(entries))
        env = os.environ.copy()
        # Academic acceleration was tested first; direct GCS was substantially
        # faster. Only this subprocess bypasses proxies; do not modify system env.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(key, None)
        subprocess.run(["aria2c", "-c", "-j", "3", "-x", "16", "-s", "16", "--min-split-size=1M",
                        "--file-allocation=none", "--summary-interval=15", "--console-log-level=warn",
                        "--max-tries=3", "--connect-timeout=15", "--timeout=30",
                        "--allow-overwrite=true", "--auto-file-renaming=false", "-i", str(queue)],
                       check=True, env=env)
    result = []
    for index in indices:
        entry = cfg["shards"][index]
        path = data / entry["name"]
        ok = verify_shard(path, entry)
        result.append({**entry, "publisher_md5_matches": ok,
                       "sha256": hash_file(path).hex() if ok else None})
    write_json(base / "logs" / ("download-audit-" + "-".join(map(str, indices)) + ".json"),
               {"sources": result, "passed": all(r["publisher_md5_matches"] for r in result)})
    if not all(r["publisher_md5_matches"] for r in result):
        raise ValueError("Incomplete or corrupt source shard")


def varint(data, pos):
    value, shift = 0, 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("Malformed protobuf varint")


def fields(data):
    data = memoryview(data)
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(data, pos)
        elif wire == 2:
            length, pos = varint(data, pos)
            value = data[pos:pos + length]
            pos += length
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = data[pos:pos + length]
            pos += length
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if pos > len(data):
            raise ValueError("Truncated protobuf")
        yield field, wire, value


NEEDED = {"video", "segmentations", "object_coordinates", "depth",
          "metadata/video_name", "metadata/num_frames", "metadata/num_instances", "metadata/width", "metadata/height",
          "metadata/depth_range", "metadata/motion_blur", "camera/focal_length", "camera/sensor_width",
          "camera/positions", "camera/quaternions", "instances/bboxes_3d", "instances/positions",
          "instances/quaternions", "instances/is_dynamic", "instances/asset_id", "instances/category"}


def decode_feature(message):
    kind, _, content = next(fields(message))
    if kind == 1:  # BytesList
        return [bytes(value) for field, wire, value in fields(content) if field == 1 and wire == 2]
    if kind == 2:  # FloatList
        parts = [np.frombuffer(value, dtype="<f4") for field, wire, value in fields(content) if field == 1 and wire in (2, 5)]
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    if kind == 3:  # Int64List
        result = []
        for field, wire, value in fields(content):
            if field != 1:
                continue
            if wire == 0:
                result.append(value)
            elif wire == 2:
                pos = 0
                while pos < len(value):
                    item, pos = varint(value, pos)
                    result.append(item)
        return np.asarray(result, dtype=np.uint64).view(np.int64)
    raise ValueError("Unknown TF Feature kind")


def parse_example(data):
    example_features = next(value for field, wire, value in fields(data) if field == 1 and wire == 2)
    result = {}
    for field, wire, entry in fields(example_features):
        if field != 1 or wire != 2:
            continue
        parts = {number: value for number, _, value in fields(entry)}
        key = bytes(parts[1]).decode()
        if key in NEEDED:
            result[key] = decode_feature(parts[2])
    return result


def records(path):
    # The source MD5 check covers all record bytes; TFRecord CRC values are not
    # independently certified by this reader. Framing and EOF are checked.
    with path.open("rb") as f:
        while header := f.read(12):
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            length = struct.unpack("<Q", header[:8])[0]
            if not 0 < length < 256 * 1024 * 1024:
                raise ValueError("Unexpected TFRecord length")
            record = f.read(length)
            checksum = f.read(4)
            if len(record) != length or len(checksum) != 4:
                raise ValueError("Truncated TFRecord body")
            yield parse_example(record)


def decode_images(blobs, channels=None):
    result = []
    for blob in blobs:
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Image decoding failed")
        if channels == 3:
            # OpenCV uses BGR for both RGB video and uint16 local XYZ maps.
            image = image[..., :3][..., ::-1]
        result.append(image)
    return np.stack(result)


def quaternion_matrices(wxyz):
    q = np.asarray(wxyz, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    r = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                  2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                  2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


def camera_matrices(annot, count, h, w):
    positions = annot["camera/positions"].reshape(count, 3)
    rotations = quaternion_matrices(annot["camera/quaternions"].reshape(count, 4))
    c2w = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    c2w[:, :3, :3] = rotations
    c2w[:, :3, 3] = positions
    # Kubric camera convention is OpenGL; convert to x-right/y-down/z-forward.
    w2c = np.diag([1, -1, -1, 1])[None] @ np.linalg.inv(c2w)
    f = float(annot["camera/focal_length"][0]) / float(annot["camera/sensor_width"][0])
    k = np.array([[f*w, 0, w/2], [0, f*h*h/w, h/2], [0, 0, 1]], np.float64)
    # Matches Kubric's f_y=focal/sensor*H/W in normalized coordinates.
    return np.broadcast_to(k, (count, 3, 3)).copy(), w2c


def project(world, k, w2c):
    camera = np.einsum("tij,tqj->tqi", w2c[:, :3, :3], world) + w2c[:, None, :3, 3]
    raster = np.einsum("tij,tqj->tqi", k, camera)
    uv = raster[..., :2] / raster[..., 2:3]
    return uv, camera


def farthest_points(xy, count, separation):
    start = int(np.argmin(np.sum((xy - xy.mean(0))**2, axis=-1)))
    chosen, distance = [], np.full(len(xy), np.inf)
    current = start
    for _ in range(count):
        chosen.append(current)
        distance = np.minimum(distance, np.sum((xy - xy[current])**2, axis=-1))
        current = int(np.argmax(distance))
        if len(chosen) < count and distance[current] < separation**2:
            raise ValueError("Insufficient distinct separated query pixels")
    return xy[chosen].astype(np.int32)


def target_interior(mask, radius):
    # Pixels outside the image are not target pixels. OpenCV's default erosion
    # border has a different convention and can retain the rightmost pixel.
    return cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def material_tracks(object_coords, boxes):
    corners = np.array(list(itertools.product([-.5, .5], repeat=3)), np.float64)
    corner_design = np.c_[corners, np.ones(8)]
    transform = np.einsum("ab,tbc->tac", np.linalg.pinv(corner_design), boxes)
    local = object_coords.astype(np.float64) / 65535 - .5
    tracks = np.einsum("qa,tac->tqc", np.c_[local, np.ones(len(local))], transform)
    reconstructed_boxes = np.einsum("ba,tac->tbc", corner_design, transform)
    if np.max(np.abs(reconstructed_boxes - boxes)) > 1e-4:
        raise ValueError("Bounding-box affine reconstruction failed")
    return tracks


def prepare_scene(base, cfg, annot, shard_name, record_index):
    frame_count = int(annot["metadata/num_frames"][0])
    instance_count = int(annot["metadata/num_instances"][0])
    h, w = (int(annot[f"metadata/{key}"][0]) for key in ("height", "width"))
    name = annot["metadata/video_name"][0].decode()
    scene_id = "scene_" + name.replace("/", "_")
    if frame_count != 24 or (h, w) != (256, 256):
        raise ValueError("Unexpected official MOVi-F schema")
    frame_protocol = cfg["frame_protocol"]
    t0, count = frame_protocol["query_frame"], frame_protocol["frames"]
    # Defaults preserve old teacher preparation. Forecasting can query source0
    # while selecting movement through the last observed frame (not the future).
    observed_end = frame_protocol.get("selection_observed_end_index", t0)
    output_reference = frame_protocol.get("last_observed_frame", t0)
    rgb = decode_images(annot["video"], channels=3)
    segmentation = decode_images(annot["segmentations"])
    local_coords = decode_images(annot["object_coordinates"], channels=3)
    depth_encoded = decode_images(annot["depth"])
    if depth_encoded.dtype != np.uint16 or local_coords.dtype != np.uint16:
        raise ValueError("Expected uint16 depth and material coordinates")
    depth_range = annot["metadata/depth_range"].astype(np.float64)
    distance = depth_range[0] + depth_encoded.astype(np.float64) * np.diff(depth_range)[0] / 65535
    boxes = annot["instances/bboxes_3d"].reshape(instance_count, frame_count, 8, 3)
    positions = annot["instances/positions"].reshape(instance_count, frame_count, 3)
    quaternions = annot["instances/quaternions"].reshape(instance_count, frame_count, 4)
    assets = [value.decode() for value in annot["instances/asset_id"]]
    candidates = []
    for instance in range(instance_count):
        mask = (segmentation[t0] == instance + 1).astype(np.uint8)
        radius = cfg["selection"]["mask_erosion_pixels"]
        interior = target_interior(mask, radius)
        translation = float(np.max(np.linalg.norm(positions[instance, :observed_end+1] - positions[instance, 0], axis=-1)))
        q = quaternions[instance, :observed_end+1].astype(np.float64)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        rotation = float(np.max(2 * np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))))
        if int(interior.sum()) >= cfg["selection"]["minimum_interior_pixels"] and (translation > .01 or rotation > .05):
            candidates.append((int(interior.sum()), instance, interior, translation, rotation))
    if not candidates:
        return {"scene_id": scene_id, "status": "EXCLUDED", "reason": "No sufficiently visible observed-moving physical object", "shard": shard_name, "record_index": record_index}
    _, instance, interior, translation, rotation = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
    yx = np.argwhere(interior)
    xy = farthest_points(yx[:, ::-1].astype(np.float64), cfg["query_count"], cfg["selection"]["minimum_query_separation_pixels"])
    object_points = local_coords[t0, xy[:, 1], xy[:, 0]]
    tracks = material_tracks(object_points, boxes[instance])[:count]
    k, w2c = camera_matrices(annot, frame_count, h, w)
    uv, cameras = project(tracks, k[:count], w2c[:count])
    pixel_anchor = xy.astype(np.float64) + .5
    anchor_error = np.linalg.norm(uv[t0] - pixel_anchor, axis=-1)
    query_distance = distance[t0, xy[:, 1], xy[:, 0]]
    depth_relative = np.abs(np.linalg.norm(cameras[t0], axis=-1) - query_distance) / np.maximum(query_distance, 1e-8)
    passed = (np.isfinite(tracks).all() and anchor_error.max() <= cfg["cpu_audit"]["gt_t0_projection_max_pixels"]
              and np.quantile(depth_relative, .9) <= cfg["cpu_audit"]["gt_t0_depth_relative_p90"])
    summary = {"scene_id": scene_id, "status": "READY" if passed else "GT_AUDIT_FAIL", "shard": shard_name,
               "record_index": record_index, "target_instance_mask_id": instance+1, "asset_id": assets[instance],
               "query_count": len(xy), "observed_translation_scene_units": translation,
               "observed_rotation_rad": rotation, "gt_t0_pixel_anchor_error_max": float(anchor_error.max()),
               "gt_t0_pixel_anchor_error_p90": float(np.quantile(anchor_error, .9)),
               "gt_t0_depth_relative_p90": float(np.quantile(depth_relative, .9)),
               "gt_finite": bool(np.isfinite(tracks).all()), "model_forward_called": False}
    destination = base / "inputs" / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "gt_audit.json", summary)
    if not passed:
        return summary
    # Query the exact projected material point. This defines a fixed known GT
    # point identity; no output anchoring, nearest-track remapping, or future fit.
    query_uv = uv[t0].copy()
    first_camera_tracks = np.einsum("ij,tqj->tqi", w2c[0, :3, :3], tracks) + w2c[0, :3, 3]
    last_observed_tracks = np.einsum("ij,tqj->tqi", w2c[output_reference, :3, :3], tracks) + w2c[output_reference, :3, 3]
    visible = np.zeros((count, len(xy)), bool)
    for frame in range(count):
        raster = np.floor(uv[frame]).astype(int)
        inside = ((raster[:, 0] >= 0) & (raster[:, 0] < w) & (raster[:, 1] >= 0) & (raster[:, 1] < h) & (cameras[frame, :, 2] > 0))
        ids = np.flatnonzero(inside)
        x, y = raster[ids].T
        same_instance = segmentation[frame, y, x] == instance + 1
        gt_distance = np.linalg.norm(cameras[frame, ids], axis=-1)
        depth_agrees = np.abs(gt_distance-distance[frame, y, x]) < .02 * np.maximum(gt_distance, 1e-8)
        visible[frame, ids] = same_instance & depth_agrees
    (destination / "rgb").mkdir(exist_ok=True)
    for frame in range(count):
        if not cv2.imwrite(str(destination / "rgb" / f"{frame:03d}.png"), rgb[frame, ..., ::-1]):
            raise RuntimeError("RGB write failed")
    cv2.imwrite(str(destination / "target_mask_t0.png"), (segmentation[t0] == instance+1).astype(np.uint8)*255)
    np.savez_compressed(destination / "gt_tracks.npz", tracks_world=tracks, tracks_first_camera=first_camera_tracks,
                        tracks_last_observed_camera=last_observed_tracks, tracks_2d=uv, visibility=visible,
                        validity=np.ones((count, len(xy)), bool), intrinsic_gt=k[:count], extrinsic_w2c_gt=w2c[:count],
                        object_coordinates_t0_uint16=object_points, target_bboxes_world=boxes[instance, :count],
                        target_instance_mask_id=np.int32(instance+1), source_frame_indices=np.arange(count),
                        timestamps_s=np.arange(count)/frame_protocol["dataset_native_fps"])
    # Model inputs are RGB and query UV only. GT is in a separate file.
    # Official keep_ratio preprocessing does not upscale a 256x256 source.
    # It rounds both sides down to multiples of the 14px patch size: 252x252.
    processed_hw = [h//14*14, w//14*14]
    ph, pw = processed_hw
    processed_uv = (query_uv + .5) * np.array([pw/w, ph/h]) - .5
    manifest = {"scene_id": scene_id, "target_asset_id": assets[instance], "target_instance_mask_id": instance+1,
                "query_source_frame": t0, "last_observed_source_frame": output_reference,
                "source_resolution_hw": [h, w], "processed_resolution_hw": processed_hw,
                "source_frame_indices": list(range(count)), "query_model_frame_index": t0,
                "query_uv_source_pixels": query_uv.tolist(), "query_uv_normalized_01": (processed_uv/[pw-1,ph-1]).tolist(),
                "query_pixel_sample_xy": xy.tolist(), "query_identity_source": "simulator material coordinates at t0",
                "model_inputs": ["RGB", "query_uv"], "groundtruth_as_model_input": False,
                "frame_protocol": frame_protocol, "query_user_review": "pending previews; no GPU authorized"}
    write_json(destination / "queries.json", manifest)
    preview = rgb[t0, ..., ::-1].copy()
    contour, _ = cv2.findContours((segmentation[t0] == instance+1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contour, -1, (0, 255, 0), 1)
    for point, (u, v) in enumerate(query_uv):
        color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[point*179//len(xy), 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(preview, (round(float(u)), round(float(v))), 2, color, -1)
    preview = cv2.resize(preview, (768, 768), interpolation=cv2.INTER_NEAREST)
    preview = cv2.copyMakeBorder(preview, 40, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(preview, f"{scene_id} | query={t0} | object id={instance+1} | {len(xy)} points", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
    cv2.imwrite(str(destination / "query_preview.png"), preview)
    summary["future_visible_fraction"] = float(visible[t0+1:].mean())
    summary["gt_tracks_shape"] = list(tracks.shape)
    write_json(destination / "gt_audit.json", summary)
    return summary


def prepare(base, cfg, indices, schema_only=False):
    summaries, source_audits = [], []
    for index in indices:
        entry = cfg["shards"][index]
        path = base / "data" / entry["name"]
        if not verify_shard(path, entry):
            raise ValueError("Source not verified: " + entry["name"])
        source_audits.append({**entry, "publisher_md5_matches": True, "sha256": hash_file(path).hex()})
        count = 0
        for record_index, annot in enumerate(records(path)):
            count += 1
            if schema_only:
                print(json.dumps({key: {"type": type(v).__name__, "shape": list(v.shape) if isinstance(v, np.ndarray) else [len(v)],
                                        "dtype": str(v.dtype) if isinstance(v, np.ndarray) else "bytes list"}
                                  for key, v in annot.items()}, indent=2), flush=True)
                return
            result = prepare_scene(base, cfg, annot, entry["name"], record_index)
            summaries.append(result)
            write_json(base / "logs" / "gt-preparation-status.json", {"complete": False, "model_forward_called": False, "scenes": summaries})
            print(json.dumps(result), flush=True)
        if count != entry["scene_count"]:
            raise ValueError("Scene count mismatch")
    ready = [r for r in summaries if r["status"] == "READY"]
    for r in ready:
        folder = base / "inputs" / r["scene_id"]
        write_input_hashes(base, folder)
    write_json(base / "logs" / "source-audit.json", {"sources": source_audits, "publisher_checksums_match": True})
    status = {"complete": True, "dataset": cfg["dataset"], "scenes_total": len(summaries), "scenes_ready": len(ready),
              "scenes": summaries, "query_count": cfg["query_count"], "gpu_forward_authorized": False,
              "model_forward_called": False, "gt_coordinate_checks_passed": all(r["status"] != "GT_AUDIT_FAIL" for r in summaries),
              "experiment1_inference_completed": False}
    write_json(base / "logs" / "gt-preparation-status.json", status)
    previews = [cv2.imread(str(base / "inputs" / r["scene_id"] / "query_preview.png")) for r in ready]
    if previews:
        thumb_h, thumb_w = 404, 384
        rows = []
        for start in range(0, len(previews), 4):
            tiles = [cv2.resize(p, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for p in previews[start:start+4]]
            tiles.extend([np.zeros_like(tiles[0])] * (4-len(tiles)))
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(base / "query_contact_sheet.jpg"), np.vstack(rows))
    print(json.dumps(status, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["download", "prepare", "schema"], required=True)
    ap.add_argument("--shards", nargs="+", type=int, default=[0, 1, 2, 3])
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    (args.base / "logs").mkdir(parents=True, exist_ok=True)
    if args.mode == "download":
        download(args.base, cfg, args.shards)
    else:
        prepare(args.base, cfg, args.shards, schema_only=args.mode == "schema")


if __name__ == "__main__":
    main()
