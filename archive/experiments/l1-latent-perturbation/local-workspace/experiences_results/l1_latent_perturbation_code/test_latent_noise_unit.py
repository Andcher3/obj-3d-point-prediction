#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.future_l1 import (
    _apply_relative_latent_noise,
    _make_latent_noise_generator,
)


def run_on(device: torch.device) -> dict:
    state = torch.linspace(-2.0, 2.0, 2 * 8192, device=device).reshape(2, 8192)
    mask = torch.tensor([True, False], device=device)

    zero, zero_stats = _apply_relative_latent_noise(state, mask, 0.0, None, "rms")
    assert zero is state
    assert zero_stats is None

    g1 = _make_latent_noise_generator(device, 17)
    out1, stats1 = _apply_relative_latent_noise(state, mask, 0.1, g1, "rms")
    g2 = _make_latent_noise_generator(device, 17)
    out2, stats2 = _apply_relative_latent_noise(state, mask, 0.1, g2, "rms")
    g3 = _make_latent_noise_generator(device, 18)
    out3, _ = _apply_relative_latent_noise(state, mask, 0.1, g3, "rms")

    assert torch.equal(out1, out2)
    assert not torch.equal(out1[0], out3[0])
    assert torch.equal(out1[1], state[1])
    assert torch.isfinite(out1).all()
    assert stats1 == stats2
    assert abs(stats1["relative_noise_rms"] - 0.1) < 0.005

    return {
        "device": str(device),
        "zero_exact_noop": True,
        "same_seed_exact": True,
        "different_seed_changes_active_vector": True,
        "inactive_vector_unchanged": True,
        "finite": True,
        "measured_relative_noise_rms": stats1["relative_noise_rms"],
    }


def main() -> None:
    results = [run_on(torch.device("cpu"))]
    if torch.cuda.is_available():
        results.append(run_on(torch.device("cuda")))
    print(json.dumps({"status": "PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()
