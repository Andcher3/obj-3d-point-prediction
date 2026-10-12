#!/usr/bin/env python
"""GPU runner for the complete latent branch-search experiment.

The script runs initial perturbation branches, performs feedback rounds, and
stores a joint branch report.  A real trajectory decoder is intentionally an
explicit dependency: this runner records retained latent branches but does
not claim that generated text is a decoded trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.future_l1 import FutureL1_Qwen3VL, RICE_Qwen3VL
from src.train.monkey_patch_forward import (
    replace_qwen3_vl_generation_forward,
    replace_qwen3_with_mixed_modality_forward,
)
from experiments.l1_latent_perturbation.complete_scheme import (
    BranchCandidate,
    run_feedback_search,
    save_search_report,
)
from experiments.l1_latent_perturbation.run_l1_latent_perturbation import (
    ANSWER_RE,
    LATENT_TOKEN_RE,
    SYSTEM_PROMPT,
    extract_latents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question", default="What will happen next in the video?")
    parser.add_argument("--sigmas", default="0.005,0.01,0.03,0.1,0.2")
    parser.add_argument("--seeds", default="11,22")
    parser.add_argument("--branch-count", type=int, default=3)
    parser.add_argument("--feedback-rounds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--scale-mode", choices=["rms", "std", "absolute"], default="rms")
    return parser.parse_args()


def sanitize(raw: str) -> tuple[str, str]:
    visible = LATENT_TOKEN_RE.sub("", raw)
    visible = " ".join(visible.split())
    match = ANSWER_RE.search(visible)
    return visible, match.group(1).strip() if match else ""


def build_inputs(processor: Any, video_path: str, question: str, nframes: int, max_pixels: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": nframes,
                    "max_pixels": max_pixels,
                    "min_pixels": 256 * 28 * 28,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    video_metadata = None
    if video_inputs is not None:
        video_inputs, video_metadata = map(list, zip(*video_inputs))
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )
    inputs.pop("mm_token_type_ids", None)
    return inputs


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(video)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [float(value) for value in args.sigmas.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    question = (
        f"{args.question.rstrip()} Focus on the target object: "
        f"{args.target_object.strip()}."
    )

    replace_qwen3_with_mixed_modality_forward()
    replace_qwen3_vl_generation_forward()
    model = FutureL1_Qwen3VL.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        max_pixels=args.max_pixels,
        min_pixels=256 * 28 * 28,
        trust_remote_code=True,
    )
    model.config.infer_head_prefer = "render"
    model.config.fixed_latent_budget = False
    model.config.infer_latent_multiplier = 2.0

    def generate_candidates(prompt: str, round_index: int) -> list[BranchCandidate]:
        inputs = build_inputs(processor, str(video), prompt, args.nframes, args.max_pixels)
        inputs = inputs.to("cuda")
        candidates: list[BranchCandidate] = []
        specs = [(sigma, seed) for sigma in sigmas for seed in seeds]
        if round_index == 0:
            # Keep two identical-noise baselines so determinism is measured,
            # rather than inferred from a single run.
            specs = [(0.0, 0), (0.0, 0)] + specs
        for spec_index, (sigma, seed) in enumerate(specs):
            with torch.inference_mode():
                output = model.generate(
                    **{key: value.clone() if isinstance(value, torch.Tensor) else value
                       for key, value in inputs.items()},
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    output_hidden_states=False,
                    latent_noise_std=sigma,
                    latent_noise_seed=seed,
                    latent_noise_scale_mode=args.scale_mode,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            trimmed = output.sequences[0, inputs.input_ids.shape[1]:]
            raw = processor.tokenizer.decode(
                trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            visible, answer = sanitize(raw)
            latents = extract_latents(output)
            branch_id = f"r{round_index}_{spec_index:02d}_sigma_{sigma:g}_seed_{seed}"
            np.savez_compressed(out_dir / f"{branch_id}_latents.npz", latents=latents.astype(np.float16))
            (out_dir / f"{branch_id}_raw.txt").write_text(raw, encoding="utf-8")
            valid = bool(answer and "<|latent_start|>" in raw and "<|latent_end|>" in raw
                        and np.isfinite(latents).all())
            candidates.append(
                BranchCandidate(
                    branch_id=branch_id,
                    answer=answer,
                    sigma=sigma,
                    seed=seed,
                    latent=latents,
                    raw=raw,
                    valid_format=valid,
                    metadata={
                        "visible": visible,
                        "latent_path": str(out_dir / f"{branch_id}_latents.npz"),
                        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    },
                )
            )
        return candidates

    initial = generate_candidates(question, 0)
    report = run_feedback_search(
        video_path=str(video),
        question=args.question,
        target_object=args.target_object,
        initial_candidates=initial,
        generate=generate_candidates,
        branch_count=args.branch_count,
        feedback_rounds=args.feedback_rounds,
    )
    report["trajectory_decoder"] = None
    report["warning"] = (
        "No second-stage trajectory decoder was supplied; retained branches are "
        "latent/text candidates, not decoded trajectories."
    )
    save_search_report(report, str(out_dir / "complete_search_report.json"))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
