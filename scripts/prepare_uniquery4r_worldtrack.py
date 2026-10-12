"""Prepare a separate 64-frame benchmark pack, CPU only, without downloading data."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from uniquery4r_worldtrack_protocol import (SUBSETS, decode_rgb, digest, dynamic_mask,
                                           load_sequence, processed_shape, query_normalization)


def prepare_one(source, output, subset, camera_kind, trusted_pickle, uv_mode, frame_count=64):
    if output.exists():
        raise FileExistsError("Preserve existing prepared sample: " + str(output))
    before = digest(source)
    sample = load_sequence(source, camera_kind, trusted_pickle, frame_count)
    h, w = sample["source_hw"]
    # Must be checked against the actual official preprocessing before inference.
    processed_hw = processed_shape(h, w)
    normalized = query_normalization(sample["source_uv"], [h, w], processed_hw, uv_mode)
    moving = dynamic_mask(sample["gt"])
    if digest(source) != before:
        raise ValueError("Publisher sample changed while reading")
    output.mkdir(parents=True)
    rgb_dir = output / "rgb"
    rgb_dir.mkdir()
    for t, jpeg in enumerate(sample["jpeg"]):
        rgb = decode_rgb(jpeg)
        if list(rgb.shape[:2]) != [h, w]:
            raise ValueError("Source image sizes differ")
        if not cv2.imwrite(str(rgb_dir / f"{t:03d}.png"), rgb[..., ::-1]):
            raise RuntimeError("RGB write failed")
    np.savez_compressed(output / "gt.npz", tracks_first_camera=sample["gt"],
                        visibility=sample["visibility"], annotation_point_ids=sample["ids"],
                        source_frame_indices=np.arange(frame_count), dynamic_mask=moving)
    manifest = {"subset": subset, "sequence": source.stem, "source_path": str(source),
                "source_sha256": before, "camera_kind": camera_kind,
                "gt_camera_key": sample["camera_key"], "gt_coordinate_frame": "first_camera",
                "frames": frame_count, "query_frame": 0, "source_frame_indices": list(range(frame_count)),
                "available_source_frames": sample["available_frames"],
                "unit": "meters (WorldTrack release declaration; verify dataset provenance)",
                "source_hw": [h, w], "processed_hw_expected": processed_hw,
                "query_ids": sample["ids"].tolist(), "query_uv_source": sample["source_uv"].tolist(),
                "query_uv_normalized_01": normalized.tolist(), "uv_mode": uv_mode,
                "selection": sample["selection"], "dynamic_count": int(moving.sum()),
                "model_inputs": ["RGB", "query_uv"], "gt_as_model_input": False,
                "query_review_approved": False, "model_forward_called": False,
                "official_uniquery_eval_wrapper_verified": False}
    manifest["input_sha256"] = {str(p.relative_to(output)): digest(p)
                                for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    preview = cv2.imread(str(rgb_dir / "000.png"))
    for index, uv in enumerate(sample["source_uv"]):
        color = (40, 230, 255) if moving[index] else (255, 160, 40)
        cv2.circle(preview, tuple(np.rint(uv).astype(int)), 1, color, -1)
    cv2.putText(preview, f"{subset}/{source.stem} | all={len(normalized)} dyn={moving.sum()}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(output / "query_preview.png"), preview)
    return {"subset": subset, "sequence": source.stem, "status": "CPU_PREPARED",
            "query_count": len(normalized), "dynamic_count": int(moving.sum()),
            "manifest": str(output / "manifest.json"), "model_forward_called": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-kind", required=True,
                        choices=["original-worldtrack", "trackcraft-enriched"])
    parser.add_argument("--trusted-publisher-pickle", action="store_true")
    parser.add_argument("--uv-mode", choices=["open_d4rt_endpoint", "pixel_center_resize"],
                        default="open_d4rt_endpoint")
    parser.add_argument("--frames", type=int, default=64,
                        help="64 for WorldTrack; 24 only for separately labelled MOVi-F diagnostics")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--limit-per-subset", type=int, default=0,
                        help="0 = all existing sequences, nonzero = diagnostic subset")
    args = parser.parse_args()
    if args.limit_per_subset < 0:
        parser.error("limit must be >=0")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "preparation.json"
    if report_path.exists():
        raise FileExistsError("Preparation report exists; select a new output directory")
    rows = []
    for subset in args.subsets:
        files = sorted((args.data_root / subset).glob("*.npz"))
        if args.limit_per_subset:
            files = files[:args.limit_per_subset]
        if not files:
            rows.append({"subset": subset, "status": "MISSING_DATA"})
        for source in files:
            try:
                rows.append(prepare_one(source, args.output / "inputs" / subset / source.stem,
                                        subset, args.camera_kind, args.trusted_publisher_pickle,
                                        args.uv_mode, args.frames))
            except Exception as exc:
                # Retain every exclusion; do not continue as if a full run passed.
                rows.append({"subset": subset, "sequence": source.stem, "status": "FAIL",
                             "error": str(exc)})
    ready = [r for r in rows if r["status"] == "CPU_PREPARED"]
    result = {"rows": rows, "ready_count": len(ready), "model_forward_called": False,
              "gpu_authorized": False, "query_review_approved": False,
              "complete_200_sequence_data": len(ready) == 200 and all(
                  sum(r["subset"] == s for r in ready) == 50 for s in SUBSETS),
              "all_attempted_inputs_prepared": bool(ready) and len(ready) == len(rows)}
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_attempted_inputs_prepared"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
