"""CPU WorldTrack rule adapter; no model imports or automatic data downloads.

Reference: HavenFeng/St4RTrack at 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91,
dust3r/{track_eval.py,track_eval_util.py,datasets/tapvid3d.py}.
UniQuery4R's unpublished evaluation wrapper is not certified by this adapter.
"""
import hashlib
from pathlib import Path

import numpy as np

SUBSETS = ("pstudio_mini", "po_mini", "ds_mini", "adt_mini")
THRESHOLDS = (0.1, 0.3, 0.5, 1.0)
FRAMES = 64
SOURCE = 0


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metric(gt, prediction):
    gt = np.asarray(gt, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if gt.shape != prediction.shape or gt.ndim != 3 or gt.shape[-1] != 3 or not gt.size:
        raise ValueError("Expected nonempty matching (T,N,3) arrays")
    # Fail visibly rather than improve a score by discarding failed predictions.
    if not np.isfinite(gt).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite tracks; no silent finite-only filtering")
    gn = np.maximum(np.linalg.norm(gt.reshape(-1, 3), axis=-1), 1e-12)
    pn = np.maximum(np.linalg.norm(prediction.reshape(-1, 3), axis=-1), 1e-12)
    scale = float(np.median(gn) / max(float(np.median(pn)), 1e-12))
    errors = np.linalg.norm(scale * prediction - gt, axis=-1)
    fractions = [float(np.mean(errors <= threshold)) for threshold in THRESHOLDS]
    return {"scale": scale, "epe": float(errors.mean()),
            "apd_percent": float(np.mean(fractions) * 100),
            "threshold_hit_percent": [value * 100 for value in fractions],
            "point_time_count": int(errors.size)}


def dynamic_mask(gt):
    gt = np.asarray(gt, np.float64)
    if gt.ndim != 3 or gt.shape[-1] != 3 or len(gt) < 2 or not np.isfinite(gt).all():
        raise ValueError("Dynamic classification requires finite (T>=2,N,3) GT")
    cumulative_motion = np.linalg.norm(np.diff(gt, axis=0), axis=-1).sum(axis=0)
    return cumulative_motion > 0.01


def score_tracks(gt, prediction):
    all_points = metric(gt, prediction)
    moving = dynamic_mask(gt)
    # The reference recomputes scale on the dynamic subset, not all-point scale.
    dynamic = metric(gt[:, moving], prediction[:, moving]) if moving.any() else None
    return {"all_points": all_points, "dynamic_points": dynamic,
            "query_count": int(gt.shape[1]), "dynamic_query_count": int(moving.sum()),
            "dynamic_rule": "sum(norm(GT[t+1]-GT[t])) > 0.01 in GT units",
            "future_visibility_filter": False}


def select_source_queries(camera_tracks, visibility, intrinsics, height, width):
    camera_tracks = np.asarray(camera_tracks, np.float64)
    visibility = np.asarray(visibility, bool)
    intrinsics = np.asarray(intrinsics, np.float64)
    if camera_tracks.ndim != 3 or camera_tracks.shape[-1] != 3:
        raise ValueError("GT camera tracks must be (T,N,3)")
    if visibility.shape != camera_tracks.shape[:2] or intrinsics.shape != (4,):
        raise ValueError("Visibility or intrinsics shape mismatch")
    if not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
        raise ValueError("Invalid GT intrinsics")
    source = camera_tracks[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Preserve the WorldTrack loader's projection denominator.
        uv = source[:, :2] / (source[:, 2:3] + 1e-8)
        uv = uv * intrinsics[:2] + intrinsics[2:]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(uv).all(axis=1)
    nonzero_depth = np.abs(source[:, 2]) > 1e-8
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    selected = visibility[0] & finite & nonzero_depth & inside
    ids = np.flatnonzero(selected)
    if not ids.size:
        raise ValueError("No first-frame visible in-bounds queries")
    return ids, uv[ids], {
        "annotation_points": int(len(source)), "source_visible": int(visibility[0].sum()),
        "selected_points": int(len(ids)),
        "excluded_visible_points": int((visibility[0] & ~selected).sum()),
        "target_mask_filter": False, "query_subsampling": False,
        "erosion": False, "selection_based_on_predictions": False}


def first_camera_tracks(camera_tracks, extrinsics):
    tracks = np.asarray(camera_tracks, np.float64)
    extrinsics = np.asarray(extrinsics, np.float64)
    if extrinsics.shape != (len(tracks), 4, 4) or not np.isfinite(extrinsics).all():
        raise ValueError("Missing or invalid GT camera extrinsics")
    if not np.allclose(extrinsics[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Nonhomogeneous extrinsics")
    rotations = extrinsics[:, :3, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
        raise ValueError("Extrinsic rotations must be proper orthogonal matrices")
    normalized = extrinsics @ np.linalg.inv(extrinsics[0])
    c2w = np.linalg.inv(normalized)
    world = np.einsum("tij,tnj->tni", c2w[:, :3, :3], tracks)
    return world + c2w[:, None, :3, 3]


def query_normalization(uv, source_hw, processed_hw, mode):
    sh, sw = source_hw
    ph, pw = processed_hw
    uv = np.asarray(uv, np.float64)
    if mode == "open_d4rt_endpoint":
        # Public OpenD4RT sparse evaluation convention acknowledged by UniQuery4R.
        return np.clip(uv / [sw - 1, sh - 1], 0, 1)
    if mode == "pixel_center_resize":
        processed = (uv + 0.5) * [pw / sw, ph / sh] - 0.5
        normalized = processed / [pw - 1, ph - 1]
        if ((normalized < 0) | (normalized > 1)).any():
            raise ValueError("Pixel-center query lies outside model grid; do not silently clip")
        return normalized
    raise ValueError("Unknown query normalization; do not guess evaluation convention")


def processed_shape(height, width, max_size=504, patch_size=14):
    # Official UniQuery4R load_fn: floor small images, round the shorter large side.
    if width < max_size and height < max_size:
        return [max(patch_size, height // patch_size * patch_size),
                max(patch_size, width // patch_size * patch_size)]
    if width >= height:
        return [max(patch_size, round(height * max_size / width / patch_size) * patch_size),
                max_size]
    return [max_size, max(patch_size, round(width * max_size / height / patch_size) * patch_size)]


def decode_rgb(jpeg):
    import cv2
    if isinstance(jpeg, np.ndarray):
        buffer = np.asarray(jpeg, np.uint8).reshape(-1)
    else:
        buffer = np.frombuffer(bytes(jpeg), np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid RGB JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_sequence(path, camera_kind="original-worldtrack", trusted_pickle=False, frame_count=FRAMES):
    if frame_count < 2:
        raise ValueError("Need at least two native frames")
    # Object arrays are only enabled explicitly for trusted publisher files.
    with np.load(path, allow_pickle=trusted_pickle) as pack:
        required = {"images_jpeg_bytes", "tracks_XYZ", "visibility", "fx_fy_cx_cy"}
        if not required <= set(pack.files):
            raise ValueError("Missing WorldTrack fields: " + str(required - set(pack.files)))
        if camera_kind == "original-worldtrack":
            if "extrinsics_w2c_gt" in pack.files:
                raise ValueError("Enriched file detected; explicitly choose trackcraft-enriched")
            camera_key = "extrinsics_w2c"
        elif camera_kind == "trackcraft-enriched":
            camera_key = "extrinsics_w2c_gt"
        elif camera_kind == "movif-simulator":
            camera_key = "extrinsics_w2c_gt"
        else:
            raise ValueError("Unknown camera provenance")
        if camera_key not in pack.files:
            raise ValueError("GT extrinsics missing; identity fallback is not allowed")
        tracks = np.asarray(pack["tracks_XYZ"])
        visible = np.asarray(pack["visibility"])
        jpeg = np.asarray(pack["images_jpeg_bytes"])
        extrinsics = np.asarray(pack[camera_key])
        lengths = [len(jpeg), len(tracks), len(visible), len(extrinsics)]
        if min(lengths) < frame_count:
            raise ValueError(f"Need {frame_count} real frames; padding/interpolation forbidden: " + str(lengths))
        if len(set(lengths)) != 1:
            raise ValueError("RGB/GT/visibility/camera time lengths disagree")
        tracks, visible, jpeg = tracks[:frame_count], visible[:frame_count].astype(bool), jpeg[:frame_count]
        rgb0 = decode_rgb(jpeg[0])
        height, width = rgb0.shape[:2]
        intrinsics = np.asarray(pack["fx_fy_cx_cy"], np.float64)
        ids, uv, selection = select_source_queries(tracks, visible, intrinsics, height, width)
        gt = first_camera_tracks(tracks[:, ids], extrinsics[:frame_count])
        if not np.isfinite(gt).all():
            raise ValueError("Selected trajectories contain nonfinite GT; no silent future filtering")
        return {"gt": gt, "source_uv": uv, "ids": ids, "jpeg": jpeg,
                "source_hw": [height, width], "visibility": visible[:, ids],
                "camera_key": camera_key, "selection": selection,
                "available_frames": lengths[0]}


def aggregate(rows, require_full=False):
    grouped = {}
    for subset in SUBSETS:
        group = [r for r in rows if r["subset"] == subset]
        usable = [r["metrics"]["dynamic_points"] for r in group
                  if r["metrics"]["dynamic_points"] is not None]
        grouped[subset] = {"sequence_count": len(group),
                           "without_dynamic_points": len(group) - len(usable),
                           "dynamic_epe": float(np.mean([m["epe"] for m in usable])) if usable else None,
                           "dynamic_apd_percent": float(np.mean([m["apd_percent"] for m in usable])) if usable else None}
    complete = all(g["sequence_count"] == 50 and not g["without_dynamic_points"] for g in grouped.values())
    if require_full and not complete:
        raise ValueError("Full comparison requires four subsets x50 sequences with dynamic points")
    present = [g for g in grouped.values() if g["dynamic_epe"] is not None]
    other = {}
    for name in sorted({r["subset"] for r in rows} - set(SUBSETS)):
        group = [r for r in rows if r["subset"] == name]
        usable = [r["metrics"]["dynamic_points"] for r in group if r["metrics"]["dynamic_points"] is not None]
        other[name] = {"sequence_count": len(group), "without_dynamic_points": len(group) - len(usable),
                       "dynamic_epe": float(np.mean([m["epe"] for m in usable])) if usable else None,
                       "dynamic_apd_percent": float(np.mean([m["apd_percent"] for m in usable])) if usable else None,
                       "worldtrack_benchmark": False}
    return {"per_subset": grouped, "other_dataset_diagnostics": other, "full_200_sequence_coverage": complete,
            "four_dataset_macro": {
                "epe": float(np.mean([g["dynamic_epe"] for g in present])),
                "apd_percent": float(np.mean([g["dynamic_apd_percent"] for g in present]))
            } if len(present) == 4 else None}
