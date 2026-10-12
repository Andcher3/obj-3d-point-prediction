#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.l1_latent_perturbation.complete_scheme import (  # noqa: E402
    BranchCandidate,
    event_distance,
    run_feedback_search,
)


class FakeDecoder:
    def decode(self, candidate, *, video_path, target_object):
        return {
            "object": target_object,
            "video": video_path,
            "motion": candidate.answer,
        }


def main() -> None:
    candidates = [
        BranchCandidate("take", "The hand takes the cup.", 0.03, 11, quality=0.9),
        BranchCandidate("move", "The hand moves the cup to the left.", 0.1, 11, quality=0.8),
        BranchCandidate("place", "The hand places the cup on the table.", 0.2, 11, quality=0.7),
        BranchCandidate("invalid", "unusable", 0.2, 22, valid_format=False),
    ]
    assert event_distance(candidates[0].answer, candidates[1].answer) > 0.0
    prompts = []

    def generate(prompt, round_index):
        prompts.append(prompt)
        return [
            BranchCandidate(
                "pour",
                "The hand pours from the cup.",
                0.2,
                22,
                quality=0.95,
            )
        ]

    report = run_feedback_search(
        video_path="/tmp/video.mp4",
        question="What will happen next in the video?",
        target_object="cup",
        initial_candidates=candidates,
        generate=generate,
        branch_count=3,
        feedback_rounds=1,
        trajectory_decoder=FakeDecoder(),
    )
    assert len(prompts) == 1
    assert report["candidate_count"] == 5
    assert len(report["retained"]) == 3
    assert len(report["decoded_trajectories"]) == 3
    assert "Target object: cup" in prompts[0]
    assert all(item["trajectory"]["object"] == "cup" for item in report["decoded_trajectories"])
    print("PASS")


if __name__ == "__main__":
    main()
