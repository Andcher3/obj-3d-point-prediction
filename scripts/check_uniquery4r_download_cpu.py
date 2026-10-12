"""Stream-check official weight integrity; defer full loading in no-card mode."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from uniquery4r.models.uniquery4r import UniQuery4R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    asset_status = json.loads((args.base / "logs" / "asset-status.json").read_text())
    path = args.base / "weights" / "uniquery4r.pth"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    expected = asset_status["official_sha256"]
    if expected and actual != expected:
        raise ValueError("Checkpoint SHA256 mismatch")
    print(json.dumps({"stage": "CHECKPOINT_STREAMING_SHA256_PASSED", "sha256": actual}), flush=True)
    approved = json.loads((args.base / "preparation_status.json").read_text())["query_approved"]
    memory_limit_file = Path("/sys/fs/cgroup/memory.max")
    limit = memory_limit_file.read_text().strip() if memory_limit_file.exists() else "max"
    enough_memory = limit == "max" or int(limit) >= 20 * 1024**3
    model = None
    if enough_memory:
        model = UniQuery4R(chunk_size=5000).eval()
        model.load_checkpoint(str(path), strict=True, map_location="cpu", weights_only=True)
    status = {
        "status": ("CPU_PREPARATION_COMPLETE_AWAITING_GPU" if approved else "CPU_PREPARATION_COMPLETE_AWAITING_QUERY_APPROVAL_AND_GPU")
                  if enough_memory else "DOWNLOAD_ENVIRONMENT_VERIFIED_AWAITING_GPU_ENABLE",
        "hf_revision": asset_status["hf_revision"], "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": actual, "official_sha256": expected,
        "sha256_matches": actual == expected if expected else None,
        "cpu_strict_weight_load_passed": enough_memory,
        "cpu_strict_weight_load_deferred": not enough_memory,
        "cgroup_memory_limit_bytes": None if limit == "max" else int(limit),
        "parameter_count": sum(p.numel() for p in model.parameters()) if model is not None else None,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "compiled_cuda_architectures": (torch._C._cuda_getArchFlags() or "").split(),
        "query_approved": approved,
        "gpu_inference_started": False,
        "numerical_quality_evaluation_started": False,
    }
    (args.base / "logs" / "cpu-preparation-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
