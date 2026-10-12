"""SemDiD-style branch comparison and feedback search for Future-L1.

This module deliberately keeps model execution behind callbacks.  It can
therefore be unit-tested on CPU and later connected to ``model.generate`` and
an actual trajectory decoder without hiding missing capabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "could", "does", "event", "future",
    "happen", "in", "is", "likely", "next", "of", "on", "or", "the", "to",
    "video", "will", "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def event_terms(text: str, target_object: str | None = None) -> frozenset[str]:
    """Extract content terms used by the lightweight semantic comparator."""
    terms = {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }
    if target_object:
        terms.update(_TOKEN_RE.findall(target_object.lower()))
    return frozenset(terms)


def event_similarity(
    left: str,
    right: str,
    *,
    target_object: str | None = None,
) -> float:
    """Return a deterministic Jaccard similarity in ``[0, 1]``.

    This is a transparent pre-filter, not a claim of human-level semantic
    judging.  A future evaluator can replace it through ``semantic_scorer``.
    """
    left_terms = event_terms(left, target_object)
    right_terms = event_terms(right, target_object)
    if not left_terms and not right_terms:
        return 1.0
    return len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)


def event_distance(left: str, right: str, *, target_object: str | None = None) -> float:
    return 1.0 - event_similarity(left, right, target_object=target_object)


@dataclass(frozen=True)
class BranchCandidate:
    """One generated future-event branch and its optional second-stage output."""

    branch_id: str
    answer: str
    sigma: float
    seed: int
    latent: Any = None
    raw: str = ""
    valid_format: bool = True
    quality: float = 0.0
    trajectory: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchComparison:
    left_id: str
    right_id: str
    similarity: float
    distance: float
    same_event: bool


class TrajectoryDecoder(Protocol):
    """Second-stage decoder contract: latent branch -> future trajectory."""

    def decode(
        self,
        candidate: BranchCandidate,
        *,
        video_path: str,
        target_object: str,
    ) -> Any:
        ...


def compare_events(
    candidates: Sequence[BranchCandidate],
    *,
    target_object: str | None = None,
    same_event_threshold: float = 0.5,
) -> list[BranchComparison]:
    """Perform pairwise SemDiD-style event comparison for generated branches."""
    comparisons: list[BranchComparison] = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            similarity = event_similarity(
                left.answer, right.answer, target_object=target_object
            )
            comparisons.append(
                BranchComparison(
                    left_id=left.branch_id,
                    right_id=right.branch_id,
                    similarity=similarity,
                    distance=1.0 - similarity,
                    same_event=similarity >= same_event_threshold,
                )
            )
    return comparisons


def retain_joint_branches(
    candidates: Iterable[BranchCandidate],
    *,
    branch_count: int,
    target_object: str | None = None,
    semantic_scorer: Callable[[BranchCandidate], float] | None = None,
    diversity_weight: float = 0.5,
) -> list[BranchCandidate]:
    """Greedily retain high-quality, mutually distinct branches.

    Unlike selecting the top answers independently, each accepted branch is
    scored against the branches already retained.  This preserves a joint
    set of alternatives instead of collapsing to one wording variant.
    """
    pool = [candidate for candidate in candidates if candidate.valid_format]
    if branch_count <= 0 or not pool:
        return []
    scorer = semantic_scorer or (lambda candidate: candidate.quality)
    selected: list[BranchCandidate] = []
    remaining = list(pool)
    while remaining and len(selected) < branch_count:
        def joint_score(candidate: BranchCandidate) -> float:
            quality = float(scorer(candidate))
            if not selected:
                return quality
            novelty = min(
                event_distance(
                    candidate.answer,
                    chosen.answer,
                    target_object=target_object,
                )
                for chosen in selected
            )
            return (1.0 - diversity_weight) * quality + diversity_weight * novelty

        best = max(
            remaining,
            key=lambda candidate: (joint_score(candidate), candidate.branch_id),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def build_feedback_prompt(
    *,
    question: str,
    target_object: str,
    retained: Sequence[BranchCandidate],
    round_index: int,
) -> str:
    """Create the feedback query for the next generation round."""
    branch_text = "\n".join(
        f"- {candidate.branch_id}: {candidate.answer}" for candidate in retained
    )
    return (
        f"{question}\n"
        f"Target object: {target_object}\n"
        f"This is feedback-search round {round_index}. Existing candidate events:\n"
        f"{branch_text or '- none'}\n"
        "Generate a new, visually grounded candidate event for the target object. "
        "It must describe the next observable motion, not a plot summary."
    )


def run_feedback_search(
    *,
    video_path: str,
    question: str,
    target_object: str,
    initial_candidates: Sequence[BranchCandidate],
    generate: Callable[[str, int], Sequence[BranchCandidate]],
    branch_count: int = 3,
    feedback_rounds: int = 1,
    trajectory_decoder: TrajectoryDecoder | None = None,
    semantic_scorer: Callable[[BranchCandidate], float] | None = None,
    diversity_weight: float = 0.5,
) -> dict[str, Any]:
    """Run joint branch retention, feedback search, and trajectory decoding.

    ``generate(prompt, round_index)`` is the only model-specific callback.
    The function raises decoder errors instead of silently marking a branch
    as decoded, so incomplete second-stage support remains visible.
    """
    all_candidates = list(initial_candidates)
    retained = retain_joint_branches(
        all_candidates,
        branch_count=branch_count,
        target_object=target_object,
        semantic_scorer=semantic_scorer,
        diversity_weight=diversity_weight,
    )
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, feedback_rounds + 1):
        prompt = build_feedback_prompt(
            question=question,
            target_object=target_object,
            retained=retained,
            round_index=round_index,
        )
        new_candidates = list(generate(prompt, round_index))
        all_candidates.extend(new_candidates)
        retained = retain_joint_branches(
            all_candidates,
            branch_count=branch_count,
            target_object=target_object,
            semantic_scorer=semantic_scorer,
            diversity_weight=diversity_weight,
        )
        rounds.append(
            {
                "round": round_index,
                "prompt": prompt,
                "generated_branch_ids": [candidate.branch_id for candidate in new_candidates],
                "retained_branch_ids": [candidate.branch_id for candidate in retained],
            }
        )

    decoded: list[dict[str, Any]] = []
    if trajectory_decoder is not None:
        for candidate in retained:
            trajectory = trajectory_decoder.decode(
                candidate,
                video_path=video_path,
                target_object=target_object,
            )
            decoded.append(
                {"branch_id": candidate.branch_id, "trajectory": trajectory}
            )

    return {
        "video_path": video_path,
        "target_object": target_object,
        "candidate_count": len(all_candidates),
        "comparisons": [
            comparison.__dict__
            for comparison in compare_events(
                all_candidates, target_object=target_object
            )
        ],
        "retained": [candidate.branch_id for candidate in retained],
        "rounds": rounds,
        "decoded_trajectories": decoded,
    }


def save_search_report(report: Mapping[str, Any], output_path: str) -> None:
    """Persist a JSON-compatible search report without silently dropping fields."""
    import json

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, ensure_ascii=False, default=str)
