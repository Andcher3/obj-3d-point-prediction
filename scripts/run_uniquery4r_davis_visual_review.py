"""Approved DAVIS queries -> raw UniQuery4R tracks and visual review only.

Never starts inference without --gpu-enabled. No label-quality gates, training,
t0 anchoring, clipping, rigid constraints, or post-hoc coordinate registration.
"""
import argparse
import colorsys
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from uniquery4r.models.uniquery4r import UniQuery4R
from uniquery4r.utils.load_fn import load_and_preprocess_images

VIDEOS = ("car-turn", "drift-turn", "boat", "soccerball")


class SourceFrameUniQuery4R(UniQuery4R):
    """Only select source/target pairs; preserve encoder/decoder and weights."""
    query_source = 6

    def _build_pair_idx(self, num_views):
        if not 0 <= self.query_source < num_views:
            raise ValueError("Query source outside context")
        return ([(self.query_source, t) for t in range(num_views) if t != self.query_source]
                + [(t, t) for t in range(num_views)])


def processed_to_source(uv, manifest):
    h, w = manifest["processed_resolution_hw"]
    oh, ow = manifest["source_resolution_hw"]
    return (uv + 0.5) * np.array([ow / w, oh / h]) - 0.5


def correspondence_to_source(warp2d, manifest):
    h, w = manifest["processed_resolution_hw"]
    uv = (warp2d + 1.0) * 0.5 * np.array([w - 1, h - 1])
    return processed_to_source(uv, manifest)


def project_to_source(xyz, extrinsic, intrinsic, manifest):
    camera = np.einsum("tij,tqj->tqi", extrinsic[:, :3, :3], xyz)
    camera += extrinsic[:, None, :3, 3]
    homogeneous = np.einsum("tij,tqj->tqi", intrinsic, camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    uv[camera[..., 2] <= 0] = np.nan
    return processed_to_source(uv, manifest)


def draw_panel(rgb, tracks, frame, title):
    canvas = cv2.copyMakeBorder(rgb, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    height, width = rgb.shape[:2]
    for point in range(tracks.shape[1]):
        color_rgb = colorsys.hsv_to_rgb(point / tracks.shape[1], 0.8, 1)
        color = tuple(round(c * 255) for c in color_rgb[::-1])
        trail = tracks[max(0, frame - 4):frame + 1, point]
        # Leave raw coordinates unchanged. Only draw finite on-screen segments.
        for a, b in zip(trail[:-1], trail[1:]):
            if (np.isfinite([a, b]).all()
                    and (0 <= a[0] < width and 0 <= a[1] < height)
                    and (0 <= b[0] < width and 0 <= b[1] < height)):
                pa = tuple(np.rint(a + [0, 40]).astype(int))
                pb = tuple(np.rint(b + [0, 40]).astype(int))
                cv2.line(canvas, pa, pb, color, 1, cv2.LINE_AA)
        u, v = tracks[frame, point]
        if np.isfinite([u, v]).all() and 0 <= u < width and 0 <= v < height:
            center = (round(float(u)), round(float(v)) + 40)
            cv2.circle(canvas, center, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
    return canvas


def render(video_dir, paths, uv2d, uv3d, manifest, output, name):
    height, width = manifest["source_resolution_hw"]
    destination = output / (name + ".mp4")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             8, (width * 2, height + 40))
    if not writer.isOpened():
        raise RuntimeError("Could not initialize video writer")
    try:
        for frame, path in enumerate(paths):
            rgb = cv2.imread(str(path))
            if rgb is None or rgb.shape[:2] != (height, width):
                raise ValueError("Unexpected input RGB")
            source_frame = manifest["source_frame_indices"][frame]
            left = draw_panel(rgb, uv2d, frame, f"{name} | src={source_frame} | raw 2D correspondence")
            right = draw_panel(rgb, uv3d, frame, f"src={source_frame} | raw 3D projected with predicted camera")
            combined = np.hstack((left, right))
            writer.write(combined)
            if frame == manifest["query_model_frame_index"]:
                cv2.imwrite(str(output / (name + "_t0_raw.png")), combined)
    finally:
        writer.release()


def cpu_checks(base):
    approval = json.loads((base / "preparation_status.json").read_text())
    if approval["query_approved"] is not True:
        raise ValueError("Query approval required")
    summaries = []
    for video in VIDEOS:
        directory = base / "inputs" / video
        manifest = json.loads((directory / "queries.json").read_text())
        for count in (7, 22):
            pairs = SourceFrameUniQuery4R._build_pair_idx(SimpleNamespace(query_source=6), count)
            assert len(pairs) == len(set(pairs)) == 2 * count - 1
            assert all((6, t) in pairs for t in range(count))
            assert all((t, t) in pairs for t in range(count))
        paths = [str(directory / "rgb" / f"{frame:05d}.jpg")
                 for frame in manifest["source_frame_indices"]]
        images = load_and_preprocess_images(paths, max_size=504, num_workers=1)
        assert list(images.shape) == [22, 3, *manifest["processed_resolution_hw"]]
        assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
        normalized = np.array(manifest["query_uv_normalized_01"])
        restored = correspondence_to_source(normalized * 2 - 1, manifest)
        np.testing.assert_allclose(restored, manifest["query_uv_source_pixels"], atol=1e-10)
        # Unit projection: camera identity; K maps z=1 points to specified pixels.
        h, w = manifest["processed_resolution_hw"]
        synthetic = np.concatenate((normalized * [w-1, h-1], np.ones((48, 1))), axis=-1)[None]
        np.testing.assert_allclose(project_to_source(synthetic, np.eye(4)[None, :3],
                                  np.eye(3)[None], manifest)[0], restored, atol=1e-10)
        summaries.append({"video": video, "image_shape": list(images.shape),
                          "source_frame": manifest["query_source_frame_index"]})
    result = {"cpu_interface_checks_passed": True, "model_forward_called": False,
              "pair_selection_7_and_22": True, "pixel_roundtrip_and_projection_unit_test": True,
              "videos": summaries}
    (base / "logs" / "visual-interface-cpu-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--gpu-enabled", action="store_true")
    ap.add_argument("--videos", nargs="+", choices=VIDEOS, default=list(VIDEOS))
    args = ap.parse_args()
    cpu_checks(args.base)
    if args.check_only:
        return
    if not args.gpu_enabled or not torch.cuda.is_available():
        raise RuntimeError("Awaiting explicit GPU-enable confirmation and available CUDA")
    assets = json.loads((args.base / "logs" / "asset-status.json").read_text())
    preparation = json.loads((args.base / "logs" / "cpu-preparation-status.json").read_text())
    interface = json.loads((args.base / "logs" / "visual-interface-cpu-check.json").read_text())
    environment = json.loads((args.base / "logs" / "environment-without-weights.json").read_text())
    if not (preparation["sha256_matches"] is True and preparation["query_approved"] is True
            and interface["cpu_interface_checks_passed"] is True
            and environment["environment_without_weights_passed"] is True):
        raise RuntimeError("Download/environment/interface preparation incomplete")
    model = SourceFrameUniQuery4R(chunk_size=5000).eval()
    model.load_checkpoint(args.base / "weights" / "uniquery4r.pth", strict=True, weights_only=True)
    model.to("cuda")
    print(json.dumps({"stage": "GPU_VISUAL_INFERENCE", "device": torch.cuda.get_device_name(),
                      "compiled_architectures": torch.cuda.get_arch_list()}), flush=True)
    for video in args.videos:
        directory = args.base / "inputs" / video
        manifest = json.loads((directory / "queries.json").read_text())
        destination = args.base / "outputs" / video
        destination.mkdir(parents=True, exist_ok=True)
        full_paths = [directory / "rgb" / f"{frame:05d}.jpg" for frame in manifest["source_frame_indices"]]
        query = torch.tensor(manifest["query_uv_normalized_01"], dtype=torch.float32, device="cuda")
        for count in (22, 7):
            name = f"{video}_{count}frames"
            paths = full_paths[:count]
            if (destination / (name + "_raw.npz")).exists():
                raise FileExistsError("Existing output; do not silently rerun: " + name)
            images = load_and_preprocess_images(list(map(str, paths)), max_size=504).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            with torch.inference_mode():
                predictions = model(images, query_uv=query)
            torch.cuda.synchronize()
            raw = {k: v.detach().float().cpu().numpy()[0] for k, v in predictions.items()
                   if isinstance(v, torch.Tensor) and k not in ("images", "query_uv")}
            pairs = list(map(tuple, predictions["pair_idx"]))
            indices = [pairs.index((6, t)) for t in range(count)]
            tracks = raw["warp3d"][indices]
            uv2d = correspondence_to_source(raw["warp2d"][indices], manifest)
            uv3d = project_to_source(tracks, raw["extrinsic"], raw["intrinsic"], manifest)
            raw.update(pair_idx=np.array(pairs), query_uv_normalized_01=query.cpu().numpy(),
                       tracks_3d=tracks, tracks_2d_source_pixels=uv2d,
                       tracks_3d_projected_source_pixels=uv3d,
                       source_frame_indices=np.array(manifest["source_frame_indices"][:count]))
            np.savez_compressed(destination / (name + "_raw.npz"), **raw)
            metadata = {"name": name, "frame_count": count, "query_source_model_index": 6,
                        "query_source_video_index": manifest["query_source_frame_index"],
                        "git_commit": assets["git_commit"], "hf_revision": assets["hf_revision"],
                        "seconds": time.monotonic()-start,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "raw_output": True, "t0_posthoc_alignment": False,
                        "scale": "global ambiguous model scale; not meters",
                        "coordinate_frame": "context view0 camera",
                        "video_fps": 8, "video_fps_meaning": "playback only",
                        "numerical_quality_evaluation_started": False}
            (destination / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
            del predictions, images
            torch.cuda.empty_cache()
            render(directory, paths, uv2d, uv3d, manifest, destination, name)
            print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
