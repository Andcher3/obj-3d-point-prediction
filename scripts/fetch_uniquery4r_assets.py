"""Fetch official source/weights via aria2c, record immutable revisions and SHA256."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "UniQuery4R-research-preparation"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


GIT_REVISION = "95119742e61c64c27b09349035019c8cd7b41e62"
HF_REVISION = "f1143e4cc35eec118e46bc4a47a680d09125573a"
OFFICIAL_WEIGHT_SHA256 = "cd71b27bbb0c7a22de36d1eacebea84d93196a4da6296c550b1ea27f3a1c5539"
OFFICIAL_WEIGHT_BYTES = 6601806018


def download(url, path, direct=False, attempts=8, connections=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if direct:
        for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            environment.pop(key, None)
    # Sparse partial files must keep their .aria2 control file. Never delete or
    # replace them on transport errors; each invocation resumes the same file.
    command = ["aria2c", "--no-conf=true", "--continue=true", "--allow-overwrite=false",
               "--auto-file-renaming=false", "--file-allocation=none",
               f"--max-connection-per-server={connections}", f"--split={connections}", "--min-split-size=4M",
               "--max-download-limit=0", "--max-overall-download-limit=0",
               "--connect-timeout=30", "--timeout=180", "--max-tries=10",
               "--retry-wait=20", "--auto-save-interval=10", "--summary-interval=20",
               "--console-log-level=warn", "--download-result=hide",
               "--dir=" + str(path.parent), "--out=" + path.name, url]
    for attempt in range(1, attempts + 1):
        print(json.dumps({"stage": "ARIA2_RESUMABLE_ATTEMPT", "attempt": attempt,
                          "max_attempts": attempts, "connections": connections,
                          "routing": "direct_domestic_mirror" if direct else "academic_accelerator"}), flush=True)
        process = subprocess.run(command, env=environment, check=False)
        if process.returncode == 0:
            return
        # Bad invocation, file collision, checksum mismatch, or authorization
        # problems require investigation, not repeated retries.
        if process.returncode not in (1, 2, 5, 6, 7, 19, 29) or attempt == attempts:
            raise RuntimeError(f"aria2 transport failed: exit={process.returncode}; partial file retained")
        delay = min(30 * 2 ** (attempt - 1), 300)
        print(json.dumps({"stage": "TRANSPORT_BACKOFF", "exit_code": process.returncode,
                          "delay_seconds": delay, "partial_retained": True}), flush=True)
        time.sleep(delay)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--checkpoint-endpoint", choices=("official", "mirror"), default="mirror")
    ap.add_argument("--connections", type=int, choices=(4, 8, 16), default=16)
    args = ap.parse_args()
    commit = GIT_REVISION
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Invalid official Git revision")
    source = args.base / "downloads" / f"UniQuery4R-{commit}.zip"
    if not source.exists() or not zipfile.is_zipfile(source):
        print(json.dumps({"stage": "SOURCE_DOWNLOAD_ARIA2", "revision": commit}), flush=True)
        download(f"https://codeload.github.com/kosmoresearch/UniQuery4R/zip/{commit}", source, attempts=3)
    else:
        print(json.dumps({"stage": "REUSE_COMPLETED_SOURCE_ZIP", "revision": commit}), flush=True)
    with zipfile.ZipFile(source) as archive:
        prefix = f"UniQuery4R-{commit}/"
        for name in archive.namelist():
            if not name.startswith(prefix) or ".." in Path(name).parts:
                raise ValueError("Unsafe source archive path")
        if archive.testzip() is not None:
            raise ValueError("Source archive CRC failed; preserve for inspection")
        archive.extractall(args.base / "source")
    # Official metadata independently verified on 2026-10-09; pin version and
    # checksum so a mirror cannot silently substitute another model revision.
    revision = HF_REVISION
    assert re.fullmatch("[0-9a-f]{40}", revision)
    expected = OFFICIAL_WEIGHT_SHA256
    checkpoint = args.base / "weights" / "uniquery4r.pth"
    print(json.dumps({"stage": "WEIGHT_DOWNLOAD_ARIA2", "revision": revision,
                      "bytes": OFFICIAL_WEIGHT_BYTES}), flush=True)
    host = "hf-mirror.com" if args.checkpoint_endpoint == "mirror" else "huggingface.co"
    if not checkpoint.exists() or Path(str(checkpoint) + ".aria2").exists() or checkpoint.stat().st_size != OFFICIAL_WEIGHT_BYTES:
        download(f"https://{host}/Kosmo-Research/UniQuery4R/resolve/{revision}/uniquery4r.pth?download=true",
                 checkpoint, direct=args.checkpoint_endpoint == "mirror", connections=args.connections)
    actual = sha256(checkpoint)
    if actual != expected:
        raise ValueError("Official checkpoint SHA256 mismatch")
    result = {"git_commit": commit, "source_directory": str(args.base / "source" / prefix),
              "source_zip_sha256": sha256(source), "hf_revision": revision,
              "checkpoint_bytes": checkpoint.stat().st_size,
              "checkpoint_sha256": actual, "official_sha256": expected,
              "download_tool": "aria2c", "academic_acceleration": True,
              "checkpoint_endpoint": host,
              "weight_download_routing": "direct_domestic_mirror" if args.checkpoint_endpoint == "mirror" else "academic_accelerator",
              "gpu_inference_started": False}
    (args.base / "logs" / "asset-status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
