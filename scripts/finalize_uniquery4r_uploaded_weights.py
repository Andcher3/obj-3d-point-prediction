"""Verify uploaded official weights, preserve remote partials, never load a model."""
import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

EXPECTED_BYTES = 6601806018
EXPECTED_SHA256 = "cd71b27bbb0c7a22de36d1eacebea84d93196a4da6296c550b1ea27f3a1c5539"
GIT_REVISION = "95119742e61c64c27b09349035019c8cd7b41e62"
HF_REVISION = "f1143e4cc35eec118e46bc4a47a680d09125573a"


def digest(path):
    checksum = hashlib.sha256()
    total = 0
    started = time.monotonic()
    last_report = started
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            checksum.update(block)
            total += len(block)
            if time.monotonic() - last_report >= 20:
                print(json.dumps({"stage": "REMOTE_STREAMING_SHA256", "bytes": total,
                                  "total": path.stat().st_size}), flush=True)
                last_report = time.monotonic()
    return checksum.hexdigest()


def argv(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")[:-1]
    except FileNotFoundError:
        return []


def stop_old_weight_download(base):
    # Only the downloader for this exact task and its verified parent/child.
    targets = set()
    aria_pids = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        command = argv(pid)
        if not command:
            continue
        decoded = [os.fsdecode(value) for value in command]
        if "fetch_assets.py" in decoded and str(base) in decoded:
            targets.add(pid)
            parent = None
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
            parent_argv = argv(parent) if parent else []
            if any(Path(os.fsdecode(value)).name == "resume_preparation_cpu.sh" for value in parent_argv):
                targets.add(parent)
        if (Path(decoded[0]).name == "aria2c" and "--dir=" + str(base / "weights") in decoded
                and "--out=uniquery4r.pth" in decoded):
            aria_pids.add(pid)
    for pid in sorted(targets):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for pid in aria_pids:
        try:
            os.kill(pid, signal.SIGCONT)
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while any(argv(pid) for pid in targets | aria_pids):
        if time.monotonic() >= deadline:
            raise RuntimeError("Old downloader did not exit; incoming and partial weights retained")
        time.sleep(0.5)
    print(json.dumps({"stage": "OLD_TASK_DOWNLOADER_STOPPED", "pids": sorted(targets | aria_pids)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    base = args.base.resolve()
    weights = (base / "weights").resolve()
    assert weights.parent == base
    incoming = (weights / "uniquery4r.pth.incoming-verified-local").resolve()
    assert incoming.parent == weights and incoming.is_file()
    if incoming.stat().st_size != EXPECTED_BYTES or digest(incoming) != EXPECTED_SHA256:
        raise ValueError("Uploaded weight size/hash mismatch; existing remote partials unchanged")
    print("UPLOADED_OFFICIAL_SHA256_PASSED", flush=True)
    stop_old_weight_download(base)
    checkpoint = weights / "uniquery4r.pth"
    control = weights / "uniquery4r.pth.aria2"
    suffix = ".partial-preserved-after-local-upload-20261009"
    backups = [(path, path.with_name(path.name + suffix)) for path in (checkpoint, control) if path.exists()]
    if any(backup.exists() for _, backup in backups):
        raise FileExistsError("Existing partial backup; no files replaced")
    for path, backup in backups:
        path.rename(backup)
    incoming.rename(checkpoint)
    source_zip = base / "downloads" / f"UniQuery4R-{GIT_REVISION}.zip"
    asset = {"git_commit": GIT_REVISION,
             "source_directory": str(base / "source" / f"UniQuery4R-{GIT_REVISION}"),
             "source_zip_sha256": digest(source_zip), "hf_revision": HF_REVISION,
             "checkpoint_bytes": EXPECTED_BYTES, "checkpoint_sha256": EXPECTED_SHA256,
             "official_sha256": EXPECTED_SHA256, "download_tool": "aria2c",
             "academic_acceleration": True, "checkpoint_endpoint": "hf-mirror.com",
             "weight_download_routing": "user_authorized_local_download_then_verified_sftp_upload",
             "remote_partial_backups": [str(backup) for _, backup in backups],
             "gpu_inference_started": False}
    (base / "logs" / "asset-status.json").write_text(json.dumps(asset, indent=2) + "\n")
    (base / "logs" / "uploaded-weight-verification.json").write_text(json.dumps(asset, indent=2) + "\n")
    print(json.dumps(asset, indent=2), flush=True)
    print("UPLOADED_WEIGHT_INSTALLED_NO_MODEL_LOAD_NO_INFERENCE", flush=True)


if __name__ == "__main__":
    main()
