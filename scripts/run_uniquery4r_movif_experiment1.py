"""MOVi-F GT teacher-calibration inference. GPU use requires explicit flags.

Reuses the already verified UniQuery4R query-frame adapter and checkpoint.
No GT camera, depth, trajectory, or future label is passed to the model.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def input_audit(base):
    status = json.loads((base / "logs" / "gt-preparation-status.json").read_text())
    if not status["complete"] or not status["gt_coordinate_checks_passed"]:
        raise ValueError("GT preparation incomplete or failed")
    ready = [r for r in status["scenes"] if r["status"] == "READY"]
    if not ready:
        raise ValueError("No usable target scenes")
    reports = []
    for scene in ready:
        directory = base / "inputs" / scene["scene_id"]
        manifest = json.loads((directory / "queries.json").read_text())
        hashes = json.loads((directory / "sha256.json").read_text())
        for relative, expected in hashes.items():
            if digest(base / relative) != expected:
                raise ValueError("Input hash mismatch: " + relative)
        uv = np.asarray(manifest["query_uv_normalized_01"])
        query_count = scene["query_count"]
        if query_count < 48 or uv.shape != (query_count, 2) or not np.isfinite(uv).all() or not ((uv >= 0) & (uv <= 1)).all():
            raise ValueError("Bad query coordinates")
        source = np.asarray(manifest["query_uv_source_pixels"])
        ph,pw = manifest["processed_resolution_hw"]
        sh,sw = manifest["source_resolution_hw"]
        restored = (uv*[pw-1,ph-1] + .5) * [sw/pw,sh/ph] - .5
        np.testing.assert_allclose(source, restored, atol=1e-10)
        gt = np.load(directory / "gt_tracks.npz", allow_pickle=False)
        assert gt["tracks_first_camera"].shape == (22,query_count,3)
        assert gt["tracks_2d"].shape == (22,query_count,2)
        np.testing.assert_allclose(source, gt["tracks_2d"][6], atol=1e-10)
        assert manifest["model_inputs"] == ["RGB", "query_uv"]
        assert manifest["groundtruth_as_model_input"] is False
        if manifest["query_model_frame_index"] != 6 or manifest["source_frame_indices"] != list(range(22)):
            raise ValueError("Expected original 22-frame/query6 protocol")
        reports.append({"scene_id": scene["scene_id"], "query_count": query_count, "input_sha256_passed": True,
                        "uv_pixel_roundtrip_passed": True, "gt_identity_correspondence_passed": True})
    result = {"cpu_input_audit_passed": True, "scene_count": len(ready), "scenes": reports,
              "model_forward_called": False, "gt_not_model_input": True}
    (base / "logs" / "inference-input-audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--preprocess-check", action="store_true", help="CPU official RGB preprocessing only; no model load/forward")
    ap.add_argument("--gpu-enabled", action="store_true")
    ap.add_argument("--query-approved", action="store_true")
    ap.add_argument("--contexts", nargs="+", type=int, choices=(7,22), default=[22])
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--worldtrack-style-metrics", action="store_true", help="Dynamic GT subset and median-norm scale; object diagnostic only")
    args = ap.parse_args()
    ready = input_audit(args.base)
    assets = json.loads((args.deployment / "logs" / "asset-status.json").read_text())
    model_repo = args.deployment / "source" / ("UniQuery4R-" + assets["git_commit"])
    if not model_repo.is_dir():
        raise FileNotFoundError("Pinned deployed model source missing")
    sys.path.insert(0, str(model_repo))
    if args.preprocess_check:
        import torch
        # CPU-only servers expose many host cores while granting a small quota.
        # Bound tensor workers; this does not construct a model or access CUDA.
        torch.set_num_threads(1)
        from uniquery4r.utils.load_fn import load_and_preprocess_images
        preprocessing = []
        for scene in ready:
            directory = args.base / "inputs" / scene["scene_id"]
            images = load_and_preprocess_images([str(directory/"rgb"/f"{t:03d}.png") for t in range(22)], max_size=504, num_workers=1)
            manifest=json.loads((directory/"queries.json").read_text())
            if list(images.shape) != [22,3,*manifest["processed_resolution_hw"]] or not images.isfinite().all():
                raise ValueError("Official preprocessing check failed")
            preprocessing.append({"scene":scene["scene_id"],"shape":list(images.shape),"passed":True})
            del images
        (args.base/"logs"/"official-preprocessing-cpu.json").write_text(json.dumps({
            "passed":True,"scene_count":len(preprocessing),"scenes":preprocessing,
            "model_loaded":False,"model_forward_called":False})+"\n")
    if args.check_only:
        print(json.dumps({"input_audit_passed": True, "scenes": len(ready), "model_forward_called": False}))
        return
    if not args.gpu_enabled or not args.query_approved:
        raise RuntimeError("Await explicit GPU and query-preview approval; model was not imported")
    sys.path.insert(0, str(args.deployment))
    from run_visual_review import SourceFrameUniQuery4R, correspondence_to_source, project_to_source, render
    from uniquery4r.utils.load_fn import load_and_preprocess_images
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU; no inference started")
    model = SourceFrameUniQuery4R(chunk_size=5000).eval()
    model.load_checkpoint(args.deployment / "weights" / "uniquery4r.pth", strict=True, weights_only=True)
    model.to("cuda")
    selected = [s for s in ready if args.scenes is None or s["scene_id"] in args.scenes]
    if args.scenes and len(selected) != len(args.scenes):
        raise ValueError("Requested scene unavailable")
    for scene in selected:
        directory = args.base / "inputs" / scene["scene_id"]
        manifest = json.loads((directory / "queries.json").read_text())
        out = args.base / "outputs" / scene["scene_id"]
        out.mkdir(parents=True, exist_ok=True)
        query = torch.tensor(manifest["query_uv_normalized_01"], dtype=torch.float32, device="cuda")
        for count in args.contexts:
            name = f"{scene['scene_id']}_{count}frames"
            raw_path = out / (name + "_raw.npz")
            if raw_path.exists():
                raise FileExistsError("Existing result is preserved: " + str(raw_path))
            paths = [directory / "rgb" / f"{frame:03d}.png" for frame in range(count)]
            images = load_and_preprocess_images(list(map(str, paths)), max_size=504, num_workers=1).to("cuda")
            if list(images.shape) != [count,3,*manifest["processed_resolution_hw"]]:
                raise ValueError("Unexpected official preprocessing shape")
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                predictions = model(images, query_uv=query)
            torch.cuda.synchronize()
            raw = {k:v.detach().float().cpu().numpy()[0] for k,v in predictions.items()
                   if isinstance(v, torch.Tensor) and k not in ("images", "query_uv")}
            pairs = list(map(tuple, predictions["pair_idx"]))
            slots = [pairs.index((6,t)) for t in range(count)]
            tracks = raw["warp3d"][slots]
            if not np.isfinite(tracks).all():
                raise ValueError("Nonfinite teacher tracks; raw output not accepted")
            uv2d = correspondence_to_source(raw["warp2d"][slots], manifest)
            uv3d = project_to_source(tracks, raw["extrinsic"], raw["intrinsic"], manifest)
            raw.update(pair_idx=np.array(pairs), query_uv_normalized_01=query.cpu().numpy(),
                       tracks_3d=tracks, tracks_2d_source_pixels=uv2d,
                       tracks_3d_projected_source_pixels=uv3d, source_frame_indices=np.arange(count))
            np.savez_compressed(raw_path, **raw)
            metadata = {"scene_id": scene["scene_id"], "frame_count": count, "query_frame_index": 6,
                        "model_inputs": ["RGB", "query_uv"], "gt_as_model_input": False,
                        "raw_output": True, "posthoc_registration": False, "scale": "ambiguous; not scene units",
                        "coordinate_frame": "first context camera (official normalize_cameras_to_first)",
                        "git_commit": assets["git_commit"], "hf_revision": assets["hf_revision"],
                        "input_manifest_sha256": digest(directory / "queries.json"),
                        "gt_sha256": digest(directory / "gt_tracks.npz"),
                        "seconds": time.monotonic()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            (out / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2)+"\n")
            if args.worldtrack_style_metrics:
                from uniquery4r_worldtrack_protocol import score_tracks
                with np.load(directory / "gt_tracks.npz", allow_pickle=False) as gt:
                    scores = score_tracks(gt["tracks_first_camera"][:count], tracks)
                scores.update(worldtrack_exact_reproduction=False, query_frame=6,
                    frame_count=count, unit="simulator scene units; meter equivalence not established")
                (out / (name + "_worldtrack_style_metrics.json")).write_text(json.dumps(scores, indent=2)+"\n")
            del predictions, images
            torch.cuda.empty_cache()
            render(directory, paths, uv2d, uv3d, manifest, out, name)
            print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
