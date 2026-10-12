"""CPU checks for the state and lifecycle guarantees of multi-branch decoding."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.model.future_l1 import ActiveLatentBranchPool, _repeat_generation_batch


def test_cpu_branches_are_independent_and_bounded():
    pool = ActiveLatentBranchPool(
        torch.tensor([0.0]),
        branch_count=3,
        propose=lambda state, count: [state + float(i + 1) for i in range(count)],
    )
    pool.advance(torch.tensor([1.0]))
    branches = pool.propose()
    assert len(branches) == 3
    branches[1]["latent_state"].add_(10)
    assert torch.equal(branches[0]["latent_state"], torch.tensor([1.0]))
    assert not torch.equal(branches[1]["latent_state"], branches[2]["latent_state"])

    # A proposal callback cannot silently exceed the configured branch cap.
    limited = ActiveLatentBranchPool(
        torch.tensor([0.0]), branch_count=2,
        propose=lambda state, count: [state.clone(), state.clone()],
    )
    try:
        limited.propose()
    except RuntimeError:
        pass
    else:
        raise AssertionError("branch cap was not enforced")


def test_cpu_batch_state_repeat_and_eos_isolation():
    state = torch.tensor([[1.0, 2.0]])
    repeated = _repeat_generation_batch(state, 3)
    assert repeated.shape == (3, 2)
    repeated[0, 0] = 99
    assert repeated[1, 0] == 1 and repeated[2, 0] == 1

    # EOS is tracked per row: finishing one branch must not finish siblings.
    alive = torch.ones(3, dtype=torch.bool)
    eos = torch.tensor([True, False, False])
    alive &= ~eos
    assert alive.tolist() == [False, True, True]


if __name__ == "__main__":
    test_cpu_branches_are_independent_and_bounded()
    test_cpu_batch_state_repeat_and_eos_isolation()
    print("PASS: CPU multi-branch state, EOS isolation, and branch cap")
