"""Event-aware branch selection for the third Future-L1 perturbation experiment.

The production path requires bidirectional NLI. Lexical overlap is retained only
for transparent diagnostics and is never silently substituted for NLI.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
_MOTION_TERMS = {
    "approach", "carry", "chop", "clean", "close", "cut", "drop", "flip",
    "hold", "insert", "lift", "lower", "move", "open", "pick", "place",
    "pour", "pull", "push", "remove", "rotate", "set", "slide", "take",
    "tilt", "turn", "wash", "wipe",
}


def content_terms(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


def target_coverage(answer: str, aliases: Sequence[str]) -> float:
    answer_terms = content_terms(answer)
    values = []
    for alias in aliases:
        terms = content_terms(alias)
        values.append(len(answer_terms & terms) / len(terms) if terms else 0.0)
    return max(values, default=0.0)


def has_observable_motion(answer: str) -> bool:
    terms = content_terms(answer)
    for term in terms:
        variants = {term}
        if term.endswith("s"): variants.add(term[:-1])
        if term.endswith("es"): variants.add(term[:-2])
        if term.endswith("ed"): variants.update((term[:-2], term[:-1]))
        if term.endswith("ing"): variants.update((term[:-3], term[:-3] + "e"))
        if variants & _MOTION_TERMS:
            return True
    return False


@dataclass(frozen=True)
class BranchCandidate:
    branch_id: str
    answer: str
    sigma: float
    seed: int
    round_index: int
    latent: Any = None
    raw: str = ""
    valid_format: bool = True
    target_coverage: float = 0.0
    motion_specific: bool = False
    quality: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLatentBranch:
    """One live latent hypothesis maintained during autoregressive decoding."""

    branch_id: str
    latent_state: Any
    score: float = 0.0
    alive: bool = True
    history: list[Any] = field(default_factory=list)


class ActiveLatentBranchPool:
    """Maintain multiple latent hypotheses at each generation step.

    The model runner calls ``advance`` immediately after producing the next
    latent state.  ``propose`` creates sibling states before the next forward
    pass; ``select`` can prune them using an event/quality callback.  This is
    intentionally tensor-agnostic so the pool can be used with either a
    single-device tensor or a model-specific cache/state object.
    """

    def __init__(
        self,
        initial_state: Any,
        *,
        branch_count: int,
        propose: Callable[[Any, int], Sequence[Any]],
        score: Callable[[Any], float] | None = None,
    ):
        if branch_count < 1:
            raise ValueError("branch_count must be positive")
        self.branch_count = branch_count
        self._propose = propose
        self._score = score or (lambda _state: 0.0)
        self.branches = [
            ActiveLatentBranch("active_0", initial_state, float(self._score(initial_state)))
        ]

    def advance(self, state: Any) -> None:
        """Update the live parent state and retain its history."""
        if not self.branches:
            raise RuntimeError("cannot advance an empty latent branch pool")
        parent = self.branches[0]
        parent.history.append(parent.latent_state)
        parent.latent_state = state
        parent.score = float(self._score(state))

    def propose(self) -> list[ActiveLatentBranch]:
        """Create sibling latent states for the next forward step."""
        parent = self.branches[0]
        proposals = list(self._propose(parent.latent_state, self.branch_count - 1))
        if len(proposals) > self.branch_count - 1:
            raise RuntimeError("proposal callback returned too many branches")
        for index, state in enumerate(proposals, start=1):
            self.branches.append(
                ActiveLatentBranch(
                    f"active_{index}",
                    state,
                    float(self._score(state)),
                    history=list(parent.history),
                )
            )
        return list(self.branches)

    def select(self, keep: Callable[[ActiveLatentBranch], bool] | None = None) -> list[ActiveLatentBranch]:
        """Prune branches while preserving at least the best-scoring branch."""
        candidates = [branch for branch in self.branches if branch.alive]
        if keep is not None:
            candidates = [branch for branch in candidates if keep(branch)]
        if not candidates:
            candidates = [max(self.branches, key=lambda branch: branch.score)]
        candidates.sort(key=lambda branch: (-branch.score, branch.branch_id))
        self.branches = candidates[: self.branch_count]
        return list(self.branches)

    def states(self) -> list[Any]:
        return [branch.latent_state for branch in self.branches]


def assess_candidate(candidate: BranchCandidate, *, target_aliases: Sequence[str]) -> BranchCandidate:
    coverage = target_coverage(candidate.answer, target_aliases)
    motion = has_observable_motion(candidate.answer)
    concise = 1.0 if 4 <= len(content_terms(candidate.answer)) <= 35 else 0.0
    quality = 0.75 * coverage + 0.20 * float(motion) + 0.05 * concise
    return replace(candidate, target_coverage=coverage, motion_specific=motion, quality=quality)


class PairwiseEntailment(Protocol):
    model_name: str

    def entailment_probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


@dataclass(frozen=True)
class EventComparison:
    left_id: str
    right_id: str
    forward_entailment: float
    reverse_entailment: float
    equivalence_score: float
    same_event: bool


def compare_events(
    candidates: Sequence[BranchCandidate],
    *,
    entailment: PairwiseEntailment,
    threshold: float,
) -> list[EventComparison]:
    unordered = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1 :]]
    directional: list[tuple[str, str]] = []
    for left, right in unordered:
        directional.extend([(left.answer, right.answer), (right.answer, left.answer)])
    probabilities = entailment.entailment_probabilities(directional)
    if len(probabilities) != len(directional):
        raise RuntimeError("entailment scorer returned an unexpected number of scores")
    result = []
    for index, (left, right) in enumerate(unordered):
        forward, reverse = probabilities[2 * index : 2 * index + 2]
        equivalence = min(float(forward), float(reverse))
        result.append(EventComparison(left.branch_id, right.branch_id, float(forward), float(reverse), equivalence, equivalence >= threshold))
    return result


def select_unique_branches(
    candidates: Iterable[BranchCandidate],
    *,
    comparisons: Sequence[EventComparison],
    branch_count: int,
    min_quality: float,
    min_target_coverage: float,
) -> list[BranchCandidate]:
    if branch_count <= 0:
        return []
    same = {frozenset((c.left_id, c.right_id)): c.same_event for c in comparisons}
    pool = [c for c in candidates if c.valid_format and c.quality >= min_quality and c.target_coverage >= min_target_coverage and c.motion_specific]
    pool.sort(key=lambda c: (-c.quality, c.round_index, c.sigma, c.seed, c.branch_id))
    selected: list[BranchCandidate] = []
    for candidate in pool:
        if any(same.get(frozenset((candidate.branch_id, prior.branch_id)), candidate.answer == prior.answer) for prior in selected):
            continue
        selected.append(candidate)
        if len(selected) == branch_count:
            break
    return selected


def build_feedback_prompt(*, question: str, target_object: str, retained: Sequence[BranchCandidate], round_index: int) -> str:
    prior = "\n".join(f"- {c.answer}" for c in retained) or "- none accepted yet"
    return (
        f"{question.rstrip()}\nTarget object: {target_object}.\n"
        f"Search round {round_index}. Already retained future events:\n{prior}\n"
        "Predict exactly one other plausible immediate future event for the target object. "
        "The event must be visibly compatible with the video, must change the target object's motion or state in a substantively different way, "
        "and must not be a paraphrase, a sub-step, or a continuation of any retained event. "
        "Use the exact target description, avoid 'or', and do not summarize the plot or an action already completed in the observed clip."
    )


class TransformersNLI:
    def __init__(self, model_path: str, *, device: str = "cuda", batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model_name = model_path
        self.torch = torch
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval().to(self.device)
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        matches = [i for i, label in labels.items() if "entail" in label]
        if len(matches) != 1:
            raise RuntimeError(f"cannot identify entailment label from {labels}")
        self.entailment_index = matches[0]
        self.cache: dict[tuple[str, str], float] = {}

    def entailment_probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        missing = list(dict.fromkeys(pair for pair in pairs if pair not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            inputs = self.tokenizer([x[0] for x in batch], [x[1] for x in batch], padding=True, truncation=True, max_length=192, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                probs = self.model(**inputs).logits.float().softmax(-1)[:, self.entailment_index].cpu().tolist()
            self.cache.update(zip(batch, map(float, probs)))
        return [self.cache[pair] for pair in pairs]


def serialise_candidate(candidate: BranchCandidate) -> dict[str, Any]:
    value = asdict(candidate)
    value.pop("latent", None)
    return value
