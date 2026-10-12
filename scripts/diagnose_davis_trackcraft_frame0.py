#!/usr/bin/env python3
"""Read-only coordinate-chain diagnosis for direct frame-0 DAVIS queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite diagnostic values")
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k
    return np.stack((fx * xyz[..., 0] / xyz[..., 2] + cx,
                     fy * xyz[..., 1] / xyz[..., 2] + cy), axis=-1)


def unproject(uv: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k
    return np.stack(((uv[..., 0] - cx) / fx * depth,
                     (uv[..., 1] - cy) / fy * depth, depth), axis=-1)


def mask_count(mask: np.ndarray, uv: np.ndarray) -> int:
    xy = np.rint(uv).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    return int(np.sum(mask[xy[inside, 1], xy[inside, 0]]))


def context(folder: Path, name: str, query: np.ndarray, source_mask: np.ndarray) -> dict:
    npz_name = "observed7_user.npz" if name == "observed7" else "full22_user.npz"
    tracks_name = "observed7_query_tracks.npz" if name == "observed7" else "full22_query_tracks.npz"
    with np.load(folder / npz_name, allow_pickle=True) as user, np.load(folder / tracks_name) as tracks:
        depth = np.asarray(user["depth_map"][0], dtype=np.float64)
        k = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
        w2c0 = np.asarray(user["extrinsics_w2c"][0], dtype=np.float64)
        encoded = np.frombuffer(user["images_jpeg_bytes"][0], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        h, w = depth.shape
        mh, mw = (int(v) for v in tracks["model_resolution_hw"])
        scale = np.array([mw / w, mh / h], dtype=np.float64)
        k_model = k * np.array([scale[0], scale[1], scale[0], scale[1]])
        query_model = np.asarray(tracks["query_uv_model"], dtype=np.float64)
        query_index = query_model.astype(np.int64)
        if image.shape[:2] != depth.shape or query_index.shape != (48, 2):
            raise ValueError("image/depth/query shape mismatch")
        if not np.allclose(tracks["query_uv_original"], query, atol=1e-4):
            raise ValueError("query identity/order mismatch")
        if not np.all((query_index >= 0) & (query_index < [mw, mh])):
            raise ValueError("model query out of bounds")

        raw_depth = depth[query[:, 1].astype(int), query[:, 0].astype(int)]
        raw_xyz = unproject(query, raw_depth, k)
        raw_roundtrip = np.linalg.norm(project(raw_xyz, k) - query, axis=1)

        resized_depth = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize((mw, mh), Image.Resampling.NEAREST))
        z = resized_depth[query_index[:, 1], query_index[:, 0]].astype(np.float64)
        sampled_xyz = unproject(query_index.astype(np.float64), z, k_model)
        pred0 = np.asarray(tracks["trackcraft3r_tracks"][0], dtype=np.float64)
        delta = np.linalg.norm(pred0 - sampled_xyz, axis=1)
        pj_model_uv = project(sampled_xyz, k_model)
        pj_original_uv = project(sampled_xyz, k)
        pred_model_uv = project(pred0, k_model)
        pred_original_uv = project(pred0, k)
        model_to_input = pred_model_uv / scale
        pj_input_uv = pj_model_uv / scale
        to_source = np.array([source_mask.shape[1] / w, source_mask.shape[0] / h])
        target3d_diag = float(np.linalg.norm(np.ptp(sampled_xyz, axis=0)))
        target2d_bbox = np.asarray(json.loads((folder / "query_frame0_manifest.json").read_text())["bbox_xyxy"])
        target2d_diag = float(np.linalg.norm(target2d_bbox[2:] - target2d_bbox[:2]))
        return {
            "context": name,
            "source_depth_hw": [h, w], "decoded_jpeg_hw": list(image.shape[:2]),
            "model_hw": [mh, mw], "K_original_fx_fy_cx_cy": k.tolist(),
            "K_model_fx_fy_cx_cy": k_model.tolist(),
            "w2c_frame0_identity_max_abs": float(np.max(np.abs(w2c0 - np.eye(4)))),
            "query_model_vs_scaled_input_max_px": float(np.max(np.abs(query_model - query * scale))),
            "original_depth_K_roundtrip_px": stats(raw_roundtrip),
            "sampled_model_Pj0_projected_K_model_vs_integer_query_px": stats(np.linalg.norm(pj_model_uv - query_index, axis=1)),
            "sampled_model_Pj0_projected_K_original_vs_input_query_px": stats(np.linalg.norm(pj_original_uv - query, axis=1)),
            "sampled_model_Pj0_projected_K_model_rescaled_vs_input_query_px": stats(np.linalg.norm(pj_input_uv - query, axis=1)),
            "pred0_vs_sampled_Pj0_3d": stats(delta),
            "pred0_vs_sampled_Pj0_normalized_3d_bbox": stats(delta / target3d_diag),
            "pred0_projected_K_model_vs_integer_query_px": stats(np.linalg.norm(pred_model_uv - query_index, axis=1)),
            "pred0_projected_K_original_vs_input_query_px": stats(np.linalg.norm(pred_original_uv - query, axis=1)),
            "pred0_projected_K_model_rescaled_vs_input_query_px": stats(np.linalg.norm(model_to_input - query, axis=1)),
            "pred0_projected_K_original_vs_input_query_normalized_bbox": stats(np.linalg.norm(pred_original_uv - query, axis=1) / target2d_diag),
            "mask_counts": {
                "query_original": mask_count(source_mask, query * to_source),
                "sampled_Pj0_K_model_rescaled": mask_count(source_mask, pj_input_uv * to_source),
                "pred0_K_model_rescaled": mask_count(source_mask, model_to_input * to_source),
                "pred0_K_original": mask_count(source_mask, pred_original_uv * to_source),
            },
            "positive_depth_48": int(np.sum(raw_depth > 0)),
            "positive_resized_depth_48": int(np.sum(z > 0)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=["car-turn", "drift-turn"])
    args = parser.parse_args()
    out = {"protocol": "read_only_frame0_coordinate_chain_v1", "videos": {}}
    for video in args.videos:
        folder = args.root / video
        manifest = json.loads((folder / "query_frame0_manifest.json").read_text(encoding="utf-8"))
        query = np.asarray(manifest["query_sets"]["48"], dtype=np.float64)
        mask = np.asarray(Image.open(args.mask_root / video / "Annotations" / "00014.png")) == 1
        out["videos"][video] = {
            "source_mask_hw": list(mask.shape),
            "source_mask_instance_pixels": int(mask.sum()),
            "query_reference_source_frame": manifest["query_reference_source_frame_index"],
            "observed7": context(folder, "observed7", query, mask),
            "full22": context(folder, "full22", query, mask),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
