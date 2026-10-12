"""Integrity/placement checks only. No model imports or quality metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    args = ap.parse_args()
    base = args.base
    inventory = json.loads((base / "input_sha256.json").read_text())
    for relative, expected in inventory.items():
        path = base / relative
        if base.resolve() not in path.resolve().parents:
            raise ValueError("Input path outside approved bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Input hash mismatch: " + relative)
    status = json.loads((base / "preparation_status.json").read_text())
    if status["query_approved"] is not True:
        raise ValueError("User query approval not recorded")
    videos = []
    for video in ("car-turn", "drift-turn", "boat", "soccerball"):
        manifest = json.loads((base / "inputs" / video / "queries.json").read_text())
        frames = manifest["source_frame_indices"]
        t0 = 14 if video == "soccerball" else 20
        assert frames == list(range(t0 - 6, t0 + 16))
        assert manifest["observed_source_frame_indices"] == frames[:7]
        assert manifest["query_source_frame_index"] == t0
        assert manifest["query_model_frame_index"] == 6
        assert manifest["query_count"] == 48
        assert manifest["processed_resolution_hw"] == [280, 504]
        assert len(set(map(tuple, manifest["query_uv_processed_pixels"]))) == 48
        assert manifest["status"] == "USER_APPROVED_AWAITING_GPU"
        videos.append({"video": video, "frames": [frames[0], frames[-1]], "t0": t0})
    result = {"approved_input_integrity_passed": True, "verified_file_count": len(inventory),
              "query_approved": True, "videos": videos, "inference_started": False}
    (base / "logs").mkdir(exist_ok=True)
    (base / "logs" / "approved-input-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
