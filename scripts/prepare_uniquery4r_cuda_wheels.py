"""Official PyPI wheel hashes + aria2 mirror downloads; no model loading."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def api(name, version):
    request = urllib.request.Request(f"https://pypi.org/pypi/{name}/{version}/json",
                                     headers={"User-Agent": "UniQuery4R-environment-preparation"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=60) as response:
        return json.load(response)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()
    destination = args.base / "wheelhouse"
    destination.mkdir(exist_ok=True)
    torch_info = api("torch", "2.8.0")
    packages = {"torch": "2.8.0", "torchvision": "0.23.0"}
    for requirement in torch_info["info"]["requires_dist"]:
        match = re.match(r"((?:nvidia[\w-]+|triton))==([\w.]+)", requirement)
        if match and ('platform_system == "Linux"' in requirement or match[1] == "triton"):
            packages[match[1]] = match[2]
    manifest = []
    for name, version in packages.items():
        information = torch_info if name == "torch" else api(name, version)
        candidates = [f for f in information["urls"] if f["filename"].endswith(".whl")
                      and "x86_64" in f["filename"] and "linux" in f["filename"]
                      and ("cp310" in f["filename"] or "py3-none" in f["filename"])]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous Linux Python3.10 wheel for {name}: {len(candidates)}")
        item = candidates[0]
        manifest.append({"package": name, "version": version, "filename": item["filename"],
                         "size": item["size"], "sha256": item["digests"]["sha256"],
                         "official_url": item["url"],
                         "mirror_url": "https://pypi.tuna.tsinghua.edu.cn" + urllib.parse.urlparse(item["url"]).path})
    (args.base / "logs" / "cuda-wheel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "CUDA_WHEEL_METADATA", "packages": len(manifest),
                      "total_bytes": sum(m["size"] for m in manifest)}), flush=True)
    if args.metadata_only:
        return
    jobs = []
    for item in manifest:
        path = destination / item["filename"]
        if not path.exists():
            # Reuse contiguous pip partial/completed downloads only after pip was stopped.
            candidates = [p for p in (args.base / "tmp").rglob(item["filename"])
                          if 0 < p.stat().st_size <= item["size"]]
            if candidates:
                best = max(candidates, key=lambda p: p.stat().st_size)
                os.link(best, path)
                print(json.dumps({"reused_pip_wheel": item["filename"], "bytes": best.stat().st_size}), flush=True)
        if path.exists() and path.stat().st_size == item["size"] and not Path(str(path) + ".aria2").exists():
            if digest(path) != item["sha256"]:
                raise ValueError("Existing wheel SHA256 mismatch: " + item["filename"])
            continue
        jobs.append(item["mirror_url"] + "\n out=" + item["filename"] + "\n")
    jobfile = destination / "aria2-wheels.txt"
    jobfile.write_text("\n".join(jobs))
    environment = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        environment.pop(key, None)
    if jobs:
        for attempt in range(1, 5):
            command = ["aria2c", "--no-conf=true", "--continue=true", "--allow-overwrite=false",
                       "--auto-file-renaming=false", "--file-allocation=none", "--max-concurrent-downloads=3",
                       "--max-connection-per-server=8", "--split=8", "--min-split-size=4M",
                       "--connect-timeout=30", "--timeout=180", "--max-tries=10", "--retry-wait=20",
                       "--auto-save-interval=10", "--summary-interval=20", "--console-log-level=warn",
                       "--dir=" + str(destination), "--input-file=" + str(jobfile)]
            print(json.dumps({"stage": "CUDA_WHEELS_ARIA2", "attempt": attempt}), flush=True)
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode == 0:
                break
            if attempt == 4:
                raise RuntimeError("Wheel download failed; partial files retained")
            time.sleep(30 * attempt)
    for item in manifest:
        path = destination / item["filename"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("Downloaded wheel SHA256 mismatch: " + item["filename"])
    print("ALL_CUDA_WHEEL_SHA256_PASSED", flush=True)


if __name__ == "__main__":
    main()
