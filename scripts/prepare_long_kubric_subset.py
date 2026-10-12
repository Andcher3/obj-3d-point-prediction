"""Download a pinned three-file Kubric subset and inspect GT on CPU only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item, cfg, base):
    path = base / "data" / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    control = Path(str(path) + ".aria2")
    if path.exists() and not control.exists() and path.stat().st_size == item["size"]:
        if digest(path) == item["sha256"]:
            return path
        raise RuntimeError("Existing complete file has wrong SHA256; preserved")
    for host, direct in (("hf-mirror.com", True), ("huggingface.co", False)):
        env = os.environ.copy()
        if direct:
            for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                env.pop(key, None)
        url = f"https://{host}/datasets/{cfg['dataset']}/resolve/{cfg['revision']}/{item['path']}?download=true"
        command = ["aria2c", "--no-conf=true", "--continue=true", "--allow-overwrite=false",
                   "--auto-file-renaming=false", "--file-allocation=none", "--check-integrity=true",
                   "--checksum=sha-256=" + item["sha256"], "--max-connection-per-server=8", "--split=8",
                   "--min-split-size=4M", "--connect-timeout=20", "--timeout=60", "--max-tries=3",
                   "--retry-wait=5", "--auto-save-interval=10", "--summary-interval=20",
                   "--console-log-level=warn", "--download-result=hide", "--dir=" + str(path.parent),
                   "--out=" + path.name, url]
        print(json.dumps({"stage": "download", "file": item["path"], "host": host,
                          "route": "mirror_direct" if direct else "academic_accelerator"}), flush=True)
        with (base / "logs" / (path.stem + "-aria2.log")).open("a") as log:
            rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            if control.exists() or path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
                raise RuntimeError("Post-download checksum/size verification failed")
            return path
        print(json.dumps({"stage": "retry_other_endpoint", "file": item["path"], "aria2_exit": rc}), flush=True)
    raise RuntimeError("Download failed on both endpoints; partial and control file preserved")


def audit(path, base):
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError("ZIP CRC failed: " + bad)
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        required = {"images_jpeg_bytes", "world_coords", "occluded", "is_bkgd", "fx_fy_cx_cy", "extrinsics_w2c_gt", "H", "W"}
        if not required.issubset(keys):
            raise RuntimeError("Missing GT fields: " + str(required - keys))
        rgb = d["images_jpeg_bytes"]
        xyz = d["world_coords"]
        occ = d["occluded"]
        fg = ~d["is_bkgd"].astype(bool).reshape(-1)
        K = d["fx_fy_cx_cy"].astype(float).reshape(4)
        E = d["extrinsics_w2c_gt"].astype(float)
        H, W = int(d["H"]), int(d["W"])
        T = len(rgb)
        assert T >= 64 and xyz.shape == (H * W, T, 3) and occ.shape == (H * W, T)
        assert E.shape == (T, 4, 4) and np.isfinite(E).all() and np.isfinite(K).all()
        assert np.all(K[:2] > 0)
        finite = True
        for start in range(0, len(xyz), 8192):
            finite = finite and bool(np.isfinite(xyz[start:start + 8192]).all())
        if not finite:
            raise RuntimeError("Non-finite GT world coordinates")
        # First-frame material-point identity check; these indices are NOT model queries.
        ids = np.flatnonzero(fg & ~occ[:, 0].astype(bool))[::max(1, int(fg.sum()) // 512)]
        cam = xyz[ids, 0] @ E[0, :3, :3].T + E[0, :3, 3]
        positive = cam[:, 2] > 0
        proj = cam[:, :2] / cam[:, 2:3] * K[:2] + K[2:]
        source_uv = np.stack((ids % W, ids // W), axis=1)
        errors = np.linalg.norm(proj - source_uv, axis=1)[positive]
        motion = np.linalg.norm(np.diff(xyz[ids, :64], axis=1), axis=2).sum(axis=1)
        previews = base / "previews"
        previews.mkdir(exist_ok=True)
        selection = [0, 6, 21, 42, 63, T - 1]
        sheet = Image.new("RGB", (960, 2 * (185 + 24)), "white")
        video = previews / (path.stem + "-rgb-81frames.mp4")
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "480x278", "-r", "12", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        dims = set()
        frame_hashes = []
        try:
            for t, buf in enumerate(rgb):
                raw = buf.tobytes() if isinstance(buf, np.ndarray) else bytes(buf)
                frame_hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as frame:
                    im = frame.convert("RGB")
                dims.add(im.size)
                if im.size != (W, H):
                    raise RuntimeError("RGB size does not match GT pixel grid")
                encoder.stdin.write(im.resize((480, 278)).tobytes())
                if t in selection:
                    slot = selection.index(t)
                    x, y = (slot % 3) * 320, (slot // 3) * 209
                    sheet.paste(im.resize((320, 185)), (x, y + 24))
                    ImageDraw.Draw(sheet).text((x + 5, y + 5), f"{path.stem} frame {t}", fill="black")
            encoder.stdin.close()
            stderr = encoder.stderr.read().decode("utf-8", "replace")
            rc = encoder.wait()
            if rc:
                raise RuntimeError("ffmpeg preview failed: " + stderr[-500:])
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
        sheet.save(previews / (path.stem + "-contact-sheet.jpg"), quality=88)
        result = {"file": path.name, "frames": T, "images_decoded": T, "unique_jpeg_payloads": len(set(frame_hashes)),
                  "image_size": [W, H], "tracks_shape": list(xyz.shape), "gt_camera_shape": list(E.shape),
                  "gt_intrinsics": K.tolist(), "zip_crc_pass": True, "finite_gt_pass": True,
                  "foreground_points_frame0": int(fg.sum()), "identity_check_points": len(ids),
                  "gt_frame0_projection_p50_px": float(np.median(errors)) if len(errors) else None,
                  "gt_frame0_projection_p90_px": float(np.percentile(errors, 90)) if len(errors) else None,
                  "sampled_gt_path_length64_p50": float(np.median(motion)) if len(motion) else None,
                  "source_fps_verified": False, "preview_display_fps": 12,
                  "instance_ids_provided": False, "object_selection_pending": True,
                  "gpu_used": False, "model_queries_created": False}
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert cfg["gpu_authorized"] is False and cfg["query_approved"] is False
    assert re.fullmatch("[a-f0-9]{40}", cfg["revision"])
    assert all(re.fullmatch("[a-f0-9]{64}", x["sha256"]) for x in cfg["files"])
    base = Path(cfg["remote_base"])
    (base / "logs").mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    status = {"dataset": cfg["dataset"], "revision": cfg["revision"], "completed": [], "gpu_used": False,
              "stage": "download", "started_unix": time.time()}
    write_json(base / "status.json", status)
    try:
        status["download_concurrency"] = 2
        status["requested_files"] = [x["path"] for x in cfg["files"]]
        write_json(base / "status.json", status)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(download, item, cfg, base): item for item in cfg["files"]}
            for future in as_completed(pending):
                item = pending[future]
                path = future.result()
                status.update(stage="cpu_audit", current_file=item["path"])
                write_json(base / "status.json", status)
                # Audits stay sequential to keep dense GT memory below the 2GiB CPU quota.
                result = audit(path, base)
                result.update(sha256=digest(path), expected_sha256=item["sha256"], bytes=path.stat().st_size)
                status["completed"].append(result)
                write_json(base / "status.json", status)
                print(json.dumps({"stage": "sample_prepared", "result": result}), flush=True)
        status.update(stage="completed", finished_unix=time.time())
        write_json(base / "status.json", status)
    except Exception as exc:
        status.update(stage="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(base / "status.json", status)
        raise


if __name__ == "__main__":
    main()
