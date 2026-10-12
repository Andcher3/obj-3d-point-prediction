"""Load the official TrackCraft3R release from its self-contained checkpoint.

The upstream inference constructor first downloads duplicate Wan weights.  The
release checkpoint already contains the full DiT, T5, and three VAE state dicts,
so this adapter instantiates the published Wan-1.3B architecture and loads those
exact tensors directly.  Only the small official tokenizer files are external.
"""

from __future__ import annotations

import copy
import os

import torch

from diffsynth.models import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor


def checkpoint_architecture_from_pretrained(torch_dtype=torch.bfloat16, device="cpu", **_kwargs):
    tokenizer_dir = os.environ["TRACKCRAFT_TOKENIZER_DIR"]
    pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
    pipe.dit = WanModel(
        dim=1536,
        in_dim=16,
        ffn_dim=8960,
        out_dim=16,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=12,
        num_layers=30,
        has_image_input=False,
    )
    pipe.vae = WanVideoVAE(z_dim=16)
    pipe.text_encoder = WanTextEncoder()
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(tokenizer_dir)
    return pipe


class ReleaseCheckpointPredictor(WanSceneFlowPredictor):
    """WanSceneFlowPredictor that restores every module from the release file."""

    def _load_checkpoint(self, checkpoint_path):
        ckpt = load_state_dict(checkpoint_path, torch_dtype=torch.bfloat16)
        groups = {"dit": {}, "vae": {}, "vae_vis": {}, "vae_pj": {}, "text_encoder": {}}
        for key, value in ckpt.items():
            for name in ("vae_vis", "vae_pj", "text_encoder", "vae", "dit"):
                prefix = f"pipe.{name}."
                if key.startswith(prefix):
                    groups[name][key[len(prefix) :]] = value
                    break
        required = ("dit", "vae", "vae_vis", "vae_pj", "text_encoder")
        missing = [name for name in required if not groups[name]]
        if missing:
            raise RuntimeError(f"Release checkpoint lacks modules: {missing}")
        modules = {
            "dit": self.pipe.dit,
            "vae": self.pipe.vae,
            "vae_vis": self.pipe.vae_vis,
            "vae_pj": self.pipe.vae_pj,
            "text_encoder": self.pipe.text_encoder,
        }
        for name, module in modules.items():
            result = module.load_state_dict(groups[name], strict=False)
            if result.unexpected_keys or result.missing_keys:
                raise RuntimeError(
                    f"{name} mismatch: unexpected={result.unexpected_keys[:8]}, "
                    f"missing={result.missing_keys[:8]}"
                )
            print(f"  release {name}: loaded {len(groups[name])} tensors")


def install_release_loader() -> None:
    WanVideoPipeline.from_pretrained = staticmethod(checkpoint_architecture_from_pretrained)

