"""64-frame UniQuery4R benchmark entry. Default is CPU input audit, never forward."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from uniquery4r_worldtrack_protocol import SUBSETS, aggregate, digest, score_tracks


def audit_inputs(base, frame_count=64):
    preparation = json.loads((base / "preparation.json").read_text())
    if not preparation["all_attempted_inputs_prepared"]:
        raise ValueError("Missing/failed data; preparation report must be resolved")
    result = []
    for row in preparation["rows"]:
        folder = base / "inputs" / row["subset"] / row["sequence"]
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["frames"] != frame_count or manifest["query_frame"] != 0:
            raise ValueError(f"Expected real {frame_count} frames and source0")
        if manifest["model_inputs"] != ["RGB", "query_uv"] or manifest["gt_as_model_input"]:
            raise ValueError("Unexpected model input or GT leakage")
        if manifest["source_frame_indices"] != list(range(frame_count)):
            raise ValueError("Frame indices must be native consecutive frames, no resampling/padding")
        for relative, expected in manifest["input_sha256"].items():
            if digest(folder / relative) != expected:
                raise ValueError("Input hash mismatch: " + relative)
        with np.load(folder / "gt.npz", allow_pickle=False) as pack:
            gt = pack["tracks_first_camera"]
            if gt.shape != (frame_count, manifest["selection"]["selected_points"], 3):
                raise ValueError("GT/query shape mismatch")
            np.testing.assert_array_equal(pack["annotation_point_ids"], manifest["query_ids"])
            if not np.isfinite(gt).all():
                raise ValueError("GT contains nonfinite values")
        uv = np.asarray(manifest["query_uv_normalized_01"])
        if uv.shape != (gt.shape[1], 2) or not np.isfinite(uv).all() or ((uv < 0) | (uv > 1)).any():
            raise ValueError("Sparse query must be finite Nx2 in [0,1]")
        result.append((folder, manifest))
    if not result:
        raise ValueError("No samples")
    return result


def select_prediction_slots(pair_idx, frame_count=64):
    pairs = [tuple(map(int, pair)) for pair in pair_idx]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate pair outputs")
    return [pairs.index((0, t)) for t in range(frame_count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--model-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preprocess-check", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--gpu-enabled", action="store_true")
    parser.add_argument("--query-approved", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    inputs = audit_inputs(args.base, args.frames)
    if args.require_full:
        if args.frames != 64:
            raise ValueError("Full WorldTrack comparison requires native64 frames")
        if len(inputs) != 200 or any(sum(m["subset"] == s for _, m in inputs) != 50 for s in SUBSETS):
            raise ValueError("Need four subsets x50 sequences before loading the model")
        if any(m["dynamic_count"] == 0 for _, m in inputs):
            raise ValueError("A sequence has no dynamic points; retain and resolve explicitly")
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if args.run_inference and not (args.gpu_enabled and args.query_approved):
        raise RuntimeError("Need explicit GPU and new query approval; model not imported")
    if args.preprocess_check or args.run_inference:
        if args.model_repo is None:
            parser.error("model-repo required")
        sys.path.insert(0, str(args.model_repo))
        # Official CPU preprocessing does not construct a model or load weights.
        from uniquery4r.utils.load_fn import load_and_preprocess_images
    checks = []
    for folder, manifest in inputs:
        if args.preprocess_check:
            images = load_and_preprocess_images(
                [str(folder / "rgb" / f"{t:03d}.png") for t in range(args.frames)],
                max_size=504, num_workers=1)
            if (list(images.shape) != [args.frames, 3, *manifest["processed_hw_expected"]]
                    or not images.isfinite().all() or images.min() < 0 or images.max() > 1):
                raise ValueError("Official preprocessing differs from prepared protocol")
            del images
        checks.append({"sequence": manifest["sequence"], "subset": manifest["subset"],
                       "queries": len(manifest["query_ids"]), "input_hashes_passed": True})
    if not args.run_inference:
        print(json.dumps({"cpu_audit_passed": True, "samples": checks,
                          "preprocess_checked": args.preprocess_check,
                          "model_loaded": False, "model_forward_called": False}, indent=2))
        return
    # No future GT, true depth or true/predicted camera enters this model call.
    import torch
    from uniquery4r.models.uniquery4r import UniQuery4R
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; no model constructed")
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("Local verified checkpoint required; no automatic downloads")
    model = UniQuery4R(chunk_size=args.chunk_size).eval()
    model.load_checkpoint(args.checkpoint, strict=True, weights_only=True)
    model.to("cuda")
    checkpoint_sha256 = digest(args.checkpoint)
    results = []
    for folder, manifest in inputs:
        output = args.base / "outputs" / manifest["subset"] / manifest["sequence"]
        if output.exists():
            raise FileExistsError("Preserve existing predictions: " + str(output))
        images = load_and_preprocess_images(
            [str(folder / "rgb" / f"{t:03d}.png") for t in range(args.frames)],
            max_size=504, num_workers=1).to("cuda")
        if list(images.shape) != [args.frames, 3, *manifest["processed_hw_expected"]]:
            raise ValueError("Preprocessing shape changed")
        query = torch.tensor(manifest["query_uv_normalized_01"], device="cuda", dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            prediction = model(images, query_uv=query)
        slots = select_prediction_slots(prediction["pair_idx"], args.frames)
        tracks = prediction["warp3d"][0, slots].float().cpu().numpy()
        with np.load(folder / "gt.npz", allow_pickle=False) as pack:
            metrics = score_tracks(pack["tracks_first_camera"], tracks)
        output.mkdir(parents=True)
        raw = {key: value.detach().float().cpu().numpy() for key, value in prediction.items()
               if isinstance(value, torch.Tensor) and key not in ("images", "query_uv")}
        np.savez_compressed(output / "raw.npz", **raw, tracks_3d=tracks,
                            pair_idx=np.asarray(prediction["pair_idx"]),
                            query_ids=np.asarray(manifest["query_ids"]), source_frame_indices=np.arange(args.frames))
        result = {"subset": manifest["subset"], "sequence": manifest["sequence"],
                  "metrics": metrics, "manifest_sha256": digest(folder / "manifest.json"),
                  "metric_distance_unit": manifest["unit"], "frames": args.frames,
                  "query_uv_mode": manifest["uv_mode"],
                  "checkpoint_sha256": checkpoint_sha256, "seconds": time.monotonic() - start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "gt_as_model_input": False, "focal_fit": False,
                  "official_uniquery_eval_wrapper_verified": False}
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        del images, query, prediction
        torch.cuda.empty_cache()
    summary = aggregate(results, args.require_full)
    summary["paper_result_reproduced"] = False  # Requires analysis, never inferred by execution.
    (args.base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
