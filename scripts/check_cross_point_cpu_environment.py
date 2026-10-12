"""Import/native preprocessing checks only. Never constructs a neural network."""
import argparse
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    base = args.base.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[key] = "1"
    asset = json.loads((base / "logs/assets-complete.json").read_text())
    sys.path.insert(0, asset["source"])
    import torch
    torch.set_num_threads(1)
    from scripts.inference_model import build_model_from_ckpt
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from transformers import AutoTokenizer
    from cross_point_offline_runtime import install_offline_runtime, local_asset, text_encoder_layout
    from diffsynth.utils import ModelConfig
    from diffsynth.models.model_manager import ModelManager
    from models_pretrained.future_scene_flow.sparse_dataset import _unproject_world, _apply_rigid, _compute_pj_norm
    import cv2
    import numpy as np

    patterns = ["diffusion_pytorch_model*.safetensors", "Wan2*_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]
    paths = {p: local_asset(base, p) for p in patterns}
    restore = install_offline_runtime(base, enable_model_loading=False)
    for pattern in patterns:
        config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern=pattern)
        config.download_if_necessary()
        if config.path != paths[pattern]:
            raise ValueError("Offline routing mismatch")
    try:
        ModelManager.load_model(object(), paths[patterns[0]])
    except PermissionError:
        loading_guard_pass = True
    else:
        raise ValueError("CPU preparation allowed model loading")
    restore()
    tokenizer = AutoTokenizer.from_pretrained(paths["google/*"], local_files_only=True)
    if not tokenizer("", return_tensors="pt")["input_ids"].numel():
        raise ValueError("Empty tokenizer output")

    cfg = json.loads((base / "logs/effective-protocol.json").read_text())
    split = json.loads((base / "split-manifest.json").read_text())["splits"]
    processor = InputProcessor()
    rows = []
    for group, scenes in split.items():
        for scene in scenes:
            folder = base / "inputs" / scene
            frames = [str(folder / "rgb" / f"{i:03d}.png") for i in range(7)]
            processed, ex, intr = processor(frames, process_res=392,
                                           num_workers=1, sequential=True)
            if processed.shape != (7, 3, 392, 392):
                # Pinned processor returns N,C,H,W, despite its public docstring.
                raise ValueError("Unexpected native DA3 processor shape")
            if not torch.isfinite(processed).all() or ex is not None or intr is not None:
                raise ValueError("RGB-only preprocessing failed")
            uv = np.asarray(json.loads((folder / "queries.json").read_text())["query_uv_source_pixels"])
            mf_uv = uv * np.array([576 / 256, 320 / 256])
            back = mf_uv / np.array([576 / 256, 320 / 256])
            if np.max(np.abs(back - uv)) > 1e-10:
                raise ValueError("Query resize roundtrip failed")
            rows.append({"scene": scene, "split": group,
                         "observed_preprocess_shape": list(processed.shape),
                         "query_mapping_roundtrip_max_px": float(np.max(np.abs(back - uv)))})
    # Reuse native sparse-dataset geometry functions on a controlled array only.
    depth = np.full((16, 24), 3., dtype=np.float64)
    k = np.array([80., 90., 11.5, 7.5])
    poses = np.tile(np.eye(4), (7, 1, 1))
    poses[:, 0, 3] = np.arange(7) / 10
    maps = np.stack([_unproject_world(depth, k, pose) for pose in poses])
    last = _apply_rigid(poses[-1], maps)
    np.testing.assert_allclose(last[-1], _unproject_world(depth, k, np.eye(4)), atol=1e-12)
    norm = _compute_pj_norm(last, 80., 2., 98.)
    np.testing.assert_allclose(((last - norm.mean) / norm.scale) * norm.scale + norm.mean,
                               last, atol=1e-12)
    compiled_architectures = torch._C._cuda_getArchFlags().split()
    result = {
        "cpu_environment_pass": True, "native_preprocessing_all_160_pass": True,
        "models_imported_without_construction": ["MotionForesight loader", "DA3 API"],
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "compiled_sm_120": "sm_120" in compiled_architectures,
        "compiled_architectures": compiled_architectures,
        "versions": {name: importlib.metadata.version(name) for name in [
            "torch", "torchvision", "numpy", "transformers", "peft", "safetensors", "depth-anything-3"]},
        "offline_asset_paths": paths, "embedded_t5": text_encoder_layout(base / "weights/trackcraft3r/model.safetensors"),
        "loading_guard_pass": loading_guard_pass, "native_geometry_synthetic_checks_pass": True,
        "native_da3_inference_arguments": list(inspect.signature(DepthAnything3.inference).parameters),
        "scene_preprocessing": rows, "model_loaded": False, "forward_called": False,
        "gpu_used": False, "real_prediction_coordinate_scale_validated": False,
        "limited_inference_dependencies": True,
        "excluded_optional_features": ["DA3 GUI", "Open3D benchmarks", "xformers acceleration", "GS rendering"],
    }
    (base / "logs/cpu-environment-interface-check.json").write_text(json.dumps(result, indent=2))
    print("CPU_ENVIRONMENT_AND_160_PREPROCESS_CHECKS_PASS", flush=True)


if __name__ == "__main__":
    main()
