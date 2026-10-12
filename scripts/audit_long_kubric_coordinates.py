"""Inspect world/normalized-camera conversion using GT-only first-frame points."""
import argparse
import json
import os
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np


def sparse_first_frame_world_points(path, ids):
    """Stream the dense NPY inside ZIP; keep at most 4MiB plus selected points."""
    with zipfile.ZipFile(path) as archive, archive.open("world_coords.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NPY version")
        assert len(shape) == 3 and shape[2] == 3 and not fortran and not dtype.hasobject
        bytes_per_point = int(np.prod(shape[1:])) * dtype.itemsize
        chunk_points = max(1, 4 * 1024 * 1024 // bytes_per_point)
        output = np.empty((len(ids), 3), dtype=float)
        for start in range(0, shape[0], chunk_points):
            stop = min(start + chunk_points, shape[0])
            raw = stream.read((stop - start) * bytes_per_point)
            if len(raw) != (stop - start) * bytes_per_point:
                raise ValueError("Truncated world_coords data")
            lo, hi = np.searchsorted(ids, [start, stop])
            if hi > lo:
                block = np.frombuffer(raw, dtype=dtype).reshape(stop - start, *shape[1:])
                output[lo:hi] = block[ids[lo:hi] - start, 0]
                del block
            if hi == len(ids):
                break
        return output


def inspect(path):
    with np.load(path, allow_pickle=False) as d:
        occ = d["occluded"]
        fg = ~d["is_bkgd"].reshape(-1).astype(bool)
        W = int(d["W"])
        k = d["fx_fy_cx_cy"]
        E = d["extrinsics_w2c_gt"]
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=float)
        ids = np.flatnonzero(fg & ~occ[:, 0].astype(bool))
        ids = ids[::max(1, len(ids) // 900)]
        assert len(ids) >= 48
        points = sparse_first_frame_world_points(path, ids)
        train = np.arange(len(ids)) % 4 != 0
        trials = []
        for pixel_offset in (0.0, 0.5):
            uv = np.stack([ids % W, ids // W], axis=1).astype(float) + pixel_offset
            ok, rvec, tvec = cv2.solvePnP(points[train], uv[train], K, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                trials.append({"pixel_offset": pixel_offset, "fit_success": False})
                continue
            ok, rvec, tvec = cv2.solvePnP(points[train], uv[train], K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
            R = cv2.Rodrigues(rvec)[0]
            predicted_uv = cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2)
            errors = np.linalg.norm(predicted_uv - uv, axis=1)[~train]
            cam = points @ R.T + tvec.reshape(3)
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = R, tvec.reshape(3)
            trials.append({"pixel_offset": pixel_offset, "fit_success": bool(ok),
                           "heldout_p50_px": float(np.median(errors)), "heldout_p90_px": float(np.percentile(errors, 90)),
                           "heldout_max_px": float(errors.max()), "positive_depth_fraction": float((cam[:, 2] > 0).mean()),
                           "derived_absolute_world_to_camera0": A.tolist()})
        valid = [x for x in trials if x.get("fit_success")]
        best = min(valid, key=lambda x: x["heldout_p90_px"]) if valid else None
        return {"file": path.name, "points_checked": len(ids), "heldout_points": int((~train).sum()),
                "normalized_camera0_identity_max_error": float(np.max(np.abs(E[0] - np.eye(4)))),
                "convention_trials": trials, "lowest_error_pixel_offset": best["pixel_offset"] if best else None,
                "first_frame_coordinate_check_pass": bool(best and best["heldout_p90_px"] < 0.25 and best["positive_depth_fraction"] == 1),
                "gt_intrinsics_fixed_not_fitted": True, "gt_used_only_for_label_coordinate_diagnostic": True,
                "model_input_change": False, "original_arrays_unchanged": True,
                "future_projection_independently_verified": False,
                "future_coordinate_formula_candidate": "X_camera_t = E_relative_t @ A_world_to_camera0 @ X_world_t"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    status = json.loads((args.base / "status.json").read_text())
    assert status["stage"] == "completed" and len(status["completed"]) == 3
    reports = [inspect(args.base / "data" / "kubric_da3" / x["file"]) for x in status["completed"]]
    result = {"scope": "CPU-only first-frame GT coordinate-convention audit", "samples": reports,
              "all_first_frame_checks_pass": all(x["first_frame_coordinate_check_pass"] for x in reports),
              "gpu_used": False, "model_queries_created": False}
    (args.base / "gt-coordinate-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
