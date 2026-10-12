"""Fetch the DA3 gitlink pinned by MotionForesight; no model import or forward."""
import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

DA3_REVISION = "41736238f5bced4debf3f2a12375d2466874866d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    source = base / "source"
    source.mkdir(exist_ok=True)
    archive_path = source / ("depth-anything-3-" + DA3_REVISION + ".zip")
    url = "https://codeload.github.com/ByteDance-Seed/Depth-Anything-3/zip/" + DA3_REVISION
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    with (base / "logs/da3-source-aria2.log").open("a") as log:
        subprocess.run([
            "aria2c", "-c", "-x", "16", "-s", "16", "--max-tries=3",
            "--connect-timeout=15", "--timeout=30", "--file-allocation=none",
            "--auto-file-renaming=false", "--allow-overwrite=false",
            "--console-log-level=error", "--summary-interval=15",
            "--dir=" + str(source), "--out=" + archive_path.name, url,
        ], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if Path(str(archive_path) + ".aria2").exists():
        raise ValueError("DA3 archive download incomplete")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("DA3 archive CRC failure")
        for item in archive.infolist():
            if not (source / item.filename).resolve().is_relative_to(source):
                raise ValueError("Unsafe archive member")
        archive.extractall(source)
    destination = source / ("Depth-Anything-3-" + DA3_REVISION)
    if not (destination / "src/depth_anything_3/api.py").exists():
        raise ValueError("DA3 source entry missing")
    result = {
        "source_ready": True, "revision": DA3_REVISION,
        "revision_origin": "MotionForesight pinned gitlink external/depth-anything-3",
        "path": str(destination), "zip_bytes": archive_path.stat().st_size,
        "zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "publisher_sha256_available": False, "zip_crc_pass": True,
        "model_loaded": False, "forward_called": False, "gpu_used": False,
    }
    (base / "logs/da3-source-complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
