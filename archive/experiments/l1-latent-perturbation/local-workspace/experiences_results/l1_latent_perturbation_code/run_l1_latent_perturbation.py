#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from difflib import SequenceMatcher
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

SYSTEM_PROMPT = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.

Use the following tags to switch your thinking mode:

1. Textual Mode: <reason>Your textual reasoning process</reason>
2. Visual Mode: <|latent_start|>Your visual reasoning process<|latent_end|>

After all thinking is complete, place the final answer inside <answer>Your Final Answer</answer>.
"""

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
LATENT_TOKEN_RE = re.compile(r"<\|latent(?:_start|_end)?\|>|<\|latent\|>", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--question", required=True)
    p.add_argument(
        "--target-object",
        default="",
        help="Optional object to focus on; appended to the question and recorded in metadata.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sigmas", default="0.01,0.03,0.1")
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--baseline-repeats", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--nframes", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--force-initial-latent", action="store_true")
    p.add_argument("--scale-mode", choices=["rms", "std", "absolute"], default="rms")
    return p.parse_args()


def sanitize(raw: str) -> tuple[str, str]:
    visible = LATENT_TOKEN_RE.sub("", raw)
    visible = re.sub(r"<\|[^>]+\|>", "", visible)
    visible = re.sub(r"\s+", " ", visible).strip()
    m = ANSWER_RE.search(visible)
    answer = m.group(1).strip() if m else ""
    return visible, answer


def extract_latents(output: Any) -> np.ndarray:
    states = getattr(output, "latent_states", None)
    mask = getattr(output, "latent_mask", None)
    if states is None or mask is None:
        return np.empty((0, 0), dtype=np.float32)
    selected = states[0][mask[0].to(dtype=torch.bool)]
    if selected.numel() == 0:
        return np.empty((0, int(states.shape[-1])), dtype=np.float32)
    return selected.detach().float().cpu().numpy()


def tensor_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def latent_comparison(base: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    if base.size == 0 or other.size == 0:
        return {"aligned_vectors": 0, "mse": None, "mean_cosine": None, "relative_l2": None}
    n = min(base.shape[0], other.shape[0])
    left = base[:n].astype(np.float64)
    right = other[:n].astype(np.float64)
    diff = right - left
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denom > 1e-12
    cosine = np.sum(left * right, axis=1)[valid] / denom[valid]
    return {
        "aligned_vectors": int(n),
        "mse": float(np.mean(diff * diff)),
        "mean_cosine": float(np.mean(cosine)) if cosine.size else None,
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(left), 1e-12)),
    }


def build_inputs(
    processor: Any,
    video_path: str,
    question: str,
    nframes: int,
    max_pixels: int,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": int(nframes),
                    "max_pixels": int(max_pixels),
                    "min_pixels": 256 * 28 * 28,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    if "mm_token_type_ids" in inputs:
        inputs.pop("mm_token_type_ids")
    return inputs, text


def main() -> None:
    args = parse_args()
    if args.baseline_repeats < 2:
        raise ValueError(
            "--baseline-repeats must be at least 2 so deterministic baseline claims "
            "are supported by an actual repeat."
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(video)

    replace_qwen3_with_mixed_modality_forward()
    replace_qwen3_vl_generation_forward()
    model_cls = RICE_Qwen3VL if args.force_initial_latent else FutureL1_Qwen3VL

    load_started = time.perf_counter()
    model = model_cls.from_pretrained(
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
    load_seconds = time.perf_counter() - load_started

    question = args.question
    if args.target_object.strip():
        question = f"{question.rstrip()} Focus on the target object: {args.target_object.strip()}."
    inputs, rendered_prompt = build_inputs(
        processor,
        str(video),
        question,
        args.nframes,
        args.max_pixels,
    )
    inputs = inputs.to("cuda")

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    specs: list[tuple[str, float, int]] = []
    for i in range(args.baseline_repeats):
        specs.append((f"baseline_{i}", 0.0, i))
    for sigma in sigmas:
        if sigma <= 0:
            continue
        for seed in seeds:
            specs.append((f"sigma_{sigma:g}_seed_{seed}", sigma, seed))

    metadata = {
        "model": args.model,
        "video": str(video),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "question": question,
        "target_object": args.target_object.strip() or None,
        "rendered_prompt": rendered_prompt,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "load_seconds": load_seconds,
        "force_initial_latent": args.force_initial_latent,
        "nframes": args.nframes,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "scale_mode": args.scale_mode,
        "specs": specs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results: list[dict[str, Any]] = []
    latent_arrays: list[np.ndarray] = []
    for run_index, (name, sigma, seed) in enumerate(specs):
        torch.manual_seed(123456)
        torch.cuda.manual_seed_all(123456)
        torch.cuda.reset_peak_memory_stats()
        run_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **run_inputs,
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
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        trimmed = output.sequences[0, inputs.input_ids.shape[1] :]
        raw = processor.tokenizer.decode(
            trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        visible, answer = sanitize(raw)
        latents = extract_latents(output)
        latent_arrays.append(latents)
        np.savez_compressed(
            out_dir / f"{run_index:02d}_{name}_latents.npz",
            latents=latents.astype(np.float16),
        )
        (out_dir / f"{run_index:02d}_{name}_raw.txt").write_text(raw)
        noise_stats = getattr(output, "latent_noise_stats", None)
        result = {
            "run_index": run_index,
            "name": name,
            "sigma": sigma,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "generated_tokens": int(trimmed.numel()),
            "latent_vector_count": int(latents.shape[0]) if latents.ndim == 2 else 0,
            "latent_dim": int(latents.shape[1]) if latents.ndim == 2 and latents.size else 0,
            "latent_finite": bool(np.isfinite(latents).all()) if latents.size else False,
            "latent_sha256": tensor_sha256(latents),
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw": raw,
            "visible": visible,
            "answer": answer,
            "has_answer_tag": bool(answer),
            "has_latent_tags": "<|latent_start|>" in raw and "<|latent_end|>" in raw,
            "noise_stats": noise_stats,
        }
        results.append(result)
        (out_dir / f"{run_index:02d}_{name}.json").write_text(
            json.dumps(result, indent=2)
        )
        (out_dir / "results.partial.json").write_text(json.dumps(results, indent=2))
        print(
            f"RUN={name}; SIGMA={sigma}; SEED={seed}; "
            f"LATENTS={result['latent_vector_count']}; "
            f"ANSWER_TAG={result['has_answer_tag']}; "
            f"TIME_S={elapsed:.3f}; PEAK_GIB={result['peak_gpu_memory_gib']:.3f}",
            flush=True,
        )

    baseline = results[0]
    baseline_latents = latent_arrays[0]
    for result, latents in zip(results, latent_arrays):
        result["vs_baseline"] = {
            **latent_comparison(baseline_latents, latents),
            "raw_exact_changed": result["raw_sha256"] != baseline["raw_sha256"],
            "answer_exact_changed": result["answer"] != baseline["answer"],
            "answer_similarity": SequenceMatcher(
                None, baseline["answer"], result["answer"]
            ).ratio(),
        }

    baseline_runs = results[: args.baseline_repeats]
    zero_noise_deterministic = all(
        r["raw_sha256"] == baseline["raw_sha256"]
        and r["latent_sha256"] == baseline["latent_sha256"]
        for r in baseline_runs
    )
    nonzero = [r for r in results if r["sigma"] > 0]
    aggregate = {
        "zero_noise_deterministic": zero_noise_deterministic,
        "runs_total": len(results),
        "nonzero_runs": len(nonzero),
        "valid_format_runs": sum(
            int(r["has_answer_tag"] and r["has_latent_tags"] and r["latent_finite"])
            for r in results
        ),
        "nonzero_latent_changed_runs": sum(
            int((r["vs_baseline"]["relative_l2"] or 0.0) > 0.0) for r in nonzero
        ),
        "nonzero_raw_changed_runs": sum(
            int(r["vs_baseline"]["raw_exact_changed"]) for r in nonzero
        ),
        "nonzero_answer_changed_runs": sum(
            int(r["vs_baseline"]["answer_exact_changed"]) for r in nonzero
        ),
        "unique_nonzero_answers": len({r["answer"] for r in nonzero}),
        "all_runs_finite": all(r["latent_finite"] for r in results),
    }
    report = {"metadata": metadata, "aggregate": aggregate, "results": results}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print("AGGREGATE=" + json.dumps(aggregate, sort_keys=True))
    print("RESULT_DIR=" + str(out_dir))


if __name__ == "__main__":
    main()
