"""Audit fixed RGB/query/time inputs and package previews; no model forward."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def audit_scene(folder, cfg):
    query = json.loads((folder / "queries.json").read_text())
    protocol = cfg["frame_protocol"]
    if (query["query_source_frame"] != 0 or query["last_observed_source_frame"] != 6
            or query["groundtruth_as_model_input"]):
        raise ValueError("Incorrect query/reference/input boundary")
    if query["frame_protocol"] != protocol:
        raise ValueError("Scene/config frame protocol mismatch")
    uv = np.asarray(query["query_uv_source_pixels"], dtype=np.float64)
    if uv.shape != (48, 2) or not np.isfinite(uv).all():
        raise ValueError("Invalid query set")
    mask = cv2.imread(str(folder / "target_mask_t0.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Missing target mask")
    sample = np.asarray(query["query_pixel_sample_xy"], dtype=np.int64)
    xy = np.rint(uv).astype(np.int64)
    h, w = mask.shape
    if (np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h)
            or not np.all(mask[xy[:, 1], xy[:, 0]] > 0)):
        raise ValueError("Projected query outside target mask")
    radius = cfg["selection"]["mask_erosion_pixels"]
    interior = cv2.erode(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    if not np.all(interior[sample[:, 1], sample[:, 0]] > 0):
        raise ValueError("Query sampling not inside eroded mask")
    rows = []
    for index in range(protocol["observed_frames"]):
        path = folder / "rgb" / f"{index:03d}.png"
        im = cv2.imread(str(path))
        if im is None or im.shape[:2] != (h, w):
            raise ValueError("Invalid observed RGB")
        rows.append({"path": str(path), "sha256": sha256(path)})
    # Metadata audit only: GT is never returned in prediction-time inputs.
    with np.load(folder / "gt_tracks.npz", allow_pickle=False) as gt:
        ts = gt["timestamps_s"]
        if not np.allclose(ts, np.arange(22) / 12, atol=1e-12):
            raise ValueError("Native timestamps differ from fixed 12fps protocol")
        if gt["tracks_last_observed_camera"].shape != (22, 48, 3):
            raise ValueError("GT/query shape mismatch")
    hashes = json.loads((folder / "sha256.json").read_text())
    for path in [folder / "queries.json", folder / "target_mask_t0.png",
                 folder / "query_preview.png"]:
        key = str(path.relative_to(folder.parent.parent))
        if sha256(path) != hashes[key]:
            raise ValueError("Prepared input changed")
    for row in rows:
        key = str(Path(row["path"]).relative_to(folder.parent.parent))
        if row["sha256"] != hashes[key]:
            raise ValueError("Prepared observed image changed")
    return {"scene": folder.name, "query_count": 48,
            "query_frame": 0, "last_observed_frame": 6,
            "all_queries_inside_target": True, "sampling_inside_eroded_mask": True,
            "timestamps_s": ts.tolist(), "observed_rgb": rows,
            "query_source_sha256": sha256(folder / "queries.json"),
            "target_mask_sha256": sha256(folder / "target_mask_t0.png"),
            "preview_sha256": sha256(folder / "query_preview.png"),
            "groundtruth_used_as_prediction_input": False}


def source_contract(repo):
    model = repo / "models_pretrained/future_scene_flow/model.py"
    tree = ast.parse(model.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "FutureSceneFlowModel")
    fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    args = [a.arg for a in fn.args.args]
    if args[:3] != ["self", "rgb_obs", "pj_obs"] or any(
            name in args for name in ["fps", "timestamps", "future_rgb", "gt_tracks"]):
        raise ValueError("Unexpected native forward interface")
    recon = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "reconstruct")
    text = ast.unparse(recon)
    if "p0_t0_norm[:, :, None]" not in text:
        raise ValueError("Cannot verify frame-0 anchor")
    return {"source_sha256": sha256(model), "forward_arguments": args,
            "query_anchor": "Pj frame 0", "explicit_fps_input": False,
            "native_frame_times_are_not_checkpoint_training_timing_proof": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    base = args.base.resolve()
    cfg = json.loads(args.config.read_text())
    split = json.loads((base / "split-manifest.json").read_text())["splits"]
    scenes = sum(split.values(), [])
    if len(set(scenes)) != 160 or {k: len(v) for k, v in split.items()} != cfg["quotas"]:
        raise ValueError("Split changed")
    results = []
    for scene in scenes:
        results.append(audit_scene(base / "inputs" / scene, cfg))
    output = {"all_inputs_pass": True, "counts": {k: len(v) for k, v in split.items()},
              "scenes": results, "source_contract": source_contract(args.repo),
              "time_basis": "MOVi-F original 12fps, no interpolation or 15fps relabelling",
              "model_loaded": False, "forward_called": False, "gpu_used": False}
    (base / "logs/cpu-input-interface-audit.json").write_text(json.dumps(output, indent=2))
    preview_files = [base / (name + "-query-contact.jpg") for name in split]
    for scene in scenes:
        preview_files += [base / "inputs" / scene / "query_preview.png",
                          base / "inputs" / scene / "queries.json"]
    preview_manifest = {str(p.relative_to(base)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                        for p in preview_files}
    manifest_path = base / "logs/query-preview-files.json"
    manifest_path.write_text(json.dumps(preview_manifest, indent=2))
    archive = base / "query-previews-cpu.zip"
    if archive.exists():
        raise FileExistsError("Preserve existing preview archive")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in preview_files + [manifest_path]:
            z.write(p, str(p.relative_to(base)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Preview archive CRC failure")
    (base / "logs/query-preview-package.json").write_text(json.dumps({
        "path": str(archive), "bytes": archive.stat().st_size,
        "sha256": sha256(archive), "zip_crc_pass": True, "query_count": 160 * 48,
        "scenes": len(scenes), "future_rgb_included": False}, indent=2))
    print("ALL_160_CPU_INPUTS_AND_INTERFACE_PASS", flush=True)
    print("PREVIEW_ARCHIVE_BYTES", archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
