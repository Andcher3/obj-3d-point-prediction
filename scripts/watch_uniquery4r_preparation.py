"""Read-only preparation monitor. Prints status; never executes jobs or inference."""
import argparse
import json
import re
import time
from pathlib import Path


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def preparation_pids():
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            if any(argument.endswith(b"/resume_preparation_cpu.sh") or argument == b"resume_preparation_cpu.sh"
                   for argument in arguments[1:]):
                pids.append(int(entry.name))
        except OSError:
            continue
    return pids


def snapshot(base):
    log_path = base / "logs" / "resume-preparation.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    progress = re.findall(r"\[#[0-9a-f]+ ([^\]\r\n]+)\]", log)
    assets = read_json(base / "logs" / "asset-status.json")
    cpu = read_json(base / "logs" / "cpu-preparation-status.json")
    interface = read_json(base / "logs" / "visual-interface-cpu-check.json")
    approved = read_json(base / "logs" / "approved-input-audit.json")
    ready = bool(assets and cpu and interface and approved
                 and assets.get("checkpoint_sha256") == assets.get("official_sha256")
                 and cpu.get("cpu_strict_weight_load_passed") and cpu.get("sha256_matches")
                 and cpu.get("cuda_build") == "12.8"
                 and "sm_120" in cpu.get("compiled_cuda_architectures", [])
                 and cpu.get("query_approved")
                 and interface.get("cpu_interface_checks_passed")
                 and approved.get("approved_input_integrity_passed")
                 and "PREPARE_COMPLETE_NO_INFERENCE" in log)
    pids = preparation_pids()
    status = "READY_AWAITING_USER_GPU_ENABLE" if ready else "RUNNING" if pids else "STOPPED_NOT_READY"
    return {"monitor": "READ_ONLY_PREPARATION", "status": status,
            "progress": progress[-1] if progress else None, "preparation_pids": pids,
            "official_revision": assets.get("hf_revision") if assets else None,
            "sha256_matches": cpu.get("sha256_matches") if cpu else None,
            "cpu_strict_load": cpu.get("cpu_strict_weight_load_passed") if cpu else None,
            "cuda_build": cpu.get("cuda_build") if cpu else None,
            "cuda_architectures": cpu.get("compiled_cuda_architectures") if cpu else None,
            "query_approved": cpu.get("query_approved") if cpu else approved.get("query_approved") if approved else None,
            "input_integrity": approved.get("approved_input_integrity_passed") if approved else None,
            "interface_checks": interface.get("cpu_interface_checks_passed") if interface else None,
            "gpu_inference_started": False,
            "checks_complete": ready}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    deadline = time.monotonic() + 11 * 60 * 60
    while time.monotonic() < deadline:
        status = snapshot(args.base)
        print(json.dumps(status), flush=True)
        if status["status"] != "RUNNING":
            return
        time.sleep(60)
    print(json.dumps({"monitor": "READ_ONLY_PREPARATION", "status": "MONITOR_TIME_LIMIT"}), flush=True)


if __name__ == "__main__":
    main()
