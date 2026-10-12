#!/usr/bin/env python3
"""Verify the transferred DAVIS probe and its locally rendered previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np


VIDEOS = ("car-turn", "drift-turn", "boat", "soccerball")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def video_shape(path: Path) -> tuple[int, int, int]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path),
    ])
    stream = json.loads(raw)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    archive = root / "remote_payload.tar.gz"
    hashes = {"remote_payload.tar.gz": digest(archive.read_bytes())}
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar if m.isfile()]
        for member in members:
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            transferred = tar.extractfile(member).read()
            local = (root / member.name).read_bytes()
            if transferred != local:
                raise ValueError(f"archive/local mismatch: {member.name}")
            hashes[member.name] = digest(local)

    result = {"archive_member_files_verified": len(members), "videos": {}}
    for video in VIDEOS:
        folder = root / video
        audit = json.loads((folder / "quality_audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "query_frame0_manifest.json").read_text(encoding="utf-8"))
        query = np.asarray(manifest["query_sets"]["48"])
        assert query.shape == (48, 2)
        assert audit["status"] == "FAIL"
        assert audit["query_reference_source_frame_index"] == 14
        assert audit["output_camera_source_frame_index"] == 20
        assert audit["source_frame_indices"] == list(range(14, 36))
        assert audit["observed_source_frame_indices"] == list(range(14, 21))
        assert audit["evaluated_future_source_frame_indices"] == list(range(21, 30))
        for name, shape in (
            ("observed7_query_tracks.npz", (7, 48, 3)),
            ("full22_query_tracks.npz", (22, 48, 3)),
        ):
            with np.load(folder / name) as tracks:
                assert tracks["trackcraft3r_tracks"].shape == shape
                np.testing.assert_allclose(tracks["query_uv_original"], query, atol=1e-4)
        with np.load(folder / "motionforesight_observed_only_arrays.npz") as mf:
            assert mf["p0_query"].shape == (22, 48, 3)
            assert mf["h_output_latent"].shape == (22, 48, 16)
            assert not bool(mf["future_frames_supplied_to_model"])
            np.testing.assert_allclose(mf["query_uv_original"], query, atol=1e-4)
        with np.load(folder / "p0_h_cache_aligned.npz") as cache:
            assert cache["p0_formal9"].shape == (9, 48, 3)
            assert cache["h_output_latent_formal9"].shape == (9, 48, 16)
            assert cache["pseudo_gt_aligned"].shape == (9, 48, 3)
            assert cache["pseudo_gt_validity"].shape == (9, 48)

        vis = root / "visualizations" / video
        expected = {
            "teacher_full22_trackcraft_query_overlay.mp4": 22,
            "observed7_trackcraft_query_overlay.mp4": 7,
        }
        for name, frames in expected.items():
            path = vis / name
            assert video_shape(path) == (854, 480, frames), path
            hashes[path.relative_to(root).as_posix()] = digest(path.read_bytes())
        for name in ("frame0_source14_48_queries.png", "t0_source20_teacher_observed_query_comparison.png"):
            path = vis / name
            image = cv2.imread(str(path))
            assert image is not None and image.shape[:2] == (480, 854), path
            hashes[path.relative_to(root).as_posix()] = digest(path.read_bytes())
        vis_manifest = vis / "visualization_manifest.json"
        data = json.loads(vis_manifest.read_text(encoding="utf-8"))
        for name, expected_hash in data["product_sha256"].items():
            assert hashes[(vis / name).relative_to(root).as_posix()] == expected_hash
        assert data["preview_encode_fps_playback_only"] == 8.0
        result["videos"][video] = {
            "quality_status": audit["status"],
            "valid_query_count": audit["valid_query_count"],
            "component_count": audit["graph"]["component_count"],
            "teacher_anchor_p90_pct": 100 * audit["teacher_frame0_anchor_normalized_bbox"]["p90"],
            "registration_p90_pct": 100 * audit["registration_crossfit_normalized_3d_bbox"]["p90"],
            "registration_max_pct": 100 * audit["registration_crossfit_normalized_3d_bbox"]["max"],
            "failed_gates": [key for key, passed in audit["gates"].items() if not passed],
        }

    raw_summary = json.loads((root / "quality_summary.json").read_text(encoding="utf-8"))
    assert raw_summary["pass_count"] == 0 and not raw_summary["passed_video_ids"]
    result["all_four_quality_pass"] = False
    result["training_started"] = False
    result["remote_archive_sha256"] = hashes["remote_payload.tar.gz"]
    (root / "verification_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / "sha256_manifest.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
