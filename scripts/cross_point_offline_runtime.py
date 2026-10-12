"""Offline asset routing, without changing MotionForesight's prediction math.

Import and metadata checks are CPU-safe. Actual tensor/model loading requires
explicit installation with enable_model_loading=True after GPU authorization.
The released TrackCraft checkpoint embeds the frozen T5 encoder, so reuse those
exact tensors rather than downloading another 10.6 GiB copy.
"""
import json
import os
import struct
from pathlib import Path

TEXT_PREFIX = "pipe.text_encoder."
TEXT_MARKER = "::embedded_text_encoder"


def safetensors_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors length")
        length = struct.unpack("<Q", size_bytes)[0]
        if length > 32 * 1024**2 or length + 8 > path.stat().st_size:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    return header


def text_encoder_layout(path):
    header = safetensors_header(path)
    rows = {k[len(TEXT_PREFIX):]: v for k, v in header.items()
            if k.startswith(TEXT_PREFIX)}
    if not rows:
        raise ValueError("Embedded text encoder missing")
    return {"tensor_count": len(rows), "bytes": sum(
        x["data_offsets"][1] - x["data_offsets"][0] for x in rows.values()),
        "dtypes": sorted({x["dtype"] for x in rows.values()}),
        "shapes": {k: v["shape"] for k, v in rows.items()}}


def local_asset(base, pattern):
    base = Path(base).resolve()
    wan = base / "weights/Wan-AI/Wan2.1-T2V-1.3B"
    if pattern == "diffusion_pytorch_model*.safetensors":
        value = wan / "diffusion_pytorch_model.safetensors"
    elif pattern in ("Wan2*_VAE.pth", "Wan2.1_VAE.pth"):
        value = wan / "Wan2.1_VAE.pth"
    elif pattern == "models_t5_umt5-xxl-enc-bf16.pth":
        path = base / "weights/trackcraft3r/model.safetensors"
        text_encoder_layout(path)  # header only; no tensor read
        return str(path) + TEXT_MARKER
    elif pattern == "google/*":
        value = wan / "google/umt5-xxl"
    else:
        raise ValueError("Unprepared offline asset pattern")
    if not value.exists():
        raise FileNotFoundError(value)
    return str(value)


def install_offline_runtime(base, enable_model_loading=False):
    """Resolve all native loader requests locally; leave source files untouched."""
    import torch
    from diffsynth.utils import ModelConfig
    from diffsynth.models.model_manager import ModelManager, load_model_from_single_file
    from diffsynth.models.wan_video_text_encoder import WanTextEncoder

    base = Path(base).resolve()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[key] = "1"
    original_download = ModelConfig.download_if_necessary
    original_load = ModelManager.load_model

    def resolve(config, use_usp=False):
        if config.path is None:
            config.path = local_asset(base, config.origin_file_pattern)
        path = str(config.path).removesuffix(TEXT_MARKER)
        if not Path(path).resolve().is_relative_to(base) or not Path(path).exists():
            raise ValueError("Offline asset outside prepared directory or missing")
        config.skip_download = True

    def load(manager, file_path, model_names=None, device=None, torch_dtype=None):
        if not enable_model_loading:
            raise PermissionError("Model loading disabled during CPU preparation")
        if isinstance(file_path, str) and file_path.endswith(TEXT_MARKER):
            from safetensors import safe_open
            path = file_path.removesuffix(TEXT_MARKER)
            # Executed only after GPU authorization. Native converter and native
            # model class receive the checkpoint's original frozen T5 tensors.
            with safe_open(path, framework="pt", device="cpu") as stream:
                state = {k[len(TEXT_PREFIX):]: stream.get_tensor(k)
                         for k in stream.keys() if k.startswith(TEXT_PREFIX)}
            dtype = torch_dtype or manager.torch_dtype
            target = device or manager.device
            names, models = load_model_from_single_file(
                state, ["wan_video_text_encoder"], [WanTextEncoder],
                "civitai", dtype, target)
            expected = models[0].state_dict()
            if set(expected) != set(state) or any(
                    tuple(expected[k].shape) != tuple(state[k].shape) for k in state):
                raise ValueError("Embedded T5 state keys/shapes differ from native encoder")
            manager.model.extend(models)
            manager.model_name.extend(names)
            manager.model_path.extend([path + TEXT_MARKER] * len(models))
            return None
        return original_load(manager, file_path, model_names=model_names,
                             device=device, torch_dtype=torch_dtype)

    ModelConfig.download_if_necessary = resolve
    ModelManager.load_model = load

    def restore():
        ModelConfig.download_if_necessary = original_download
        ModelManager.load_model = original_load
    return restore
