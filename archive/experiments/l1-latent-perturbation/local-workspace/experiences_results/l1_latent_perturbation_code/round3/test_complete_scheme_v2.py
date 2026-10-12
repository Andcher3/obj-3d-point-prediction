from complete_scheme_v2 import (
    ActiveLatentBranchPool,
    BranchCandidate,
    assess_candidate,
    build_feedback_prompt,
    compare_events,
    select_unique_branches,
)


class FakeNLI:
    model_name = "fake-bidirectional-entailment"
    def entailment_probabilities(self, pairs):
        equivalent = {
            frozenset(("The hand lifts the cup.", "A person picks up the mug.")),
        }
        scores=[]
        for left,right in pairs:
            if left == right or frozenset((left,right)) in equivalent:
                scores.append(.98)
            else:
                scores.append(.03)
        return scores


def candidate(branch_id, answer, target="pink rectangular cutting board"):
    item=BranchCandidate(branch_id,answer,.1,11,0,latent=[[1]],valid_format=True)
    return assess_candidate(item,target_aliases=[target])


def main():
    pool = ActiveLatentBranchPool(
        0.0,
        branch_count=3,
        propose=lambda state, count: [state + index + 1 for index in range(count)],
        score=lambda state: -abs(state - 2.0),
    )
    pool.advance(1.0)
    assert [branch.branch_id for branch in pool.propose()] == ["active_0", "active_1", "active_2"]
    pool.select()
    assert pool.states() == [2.0, 1.0, 3.0]

    items=[
        candidate("wrong", "The blue cutting board will be moved by the hand."),
        candidate("move1", "The pink rectangular cutting board will be moved by the hand."),
        candidate("move2", "The pink rectangular cutting board will be moved by the hand."),
        candidate("place", "The pink rectangular cutting board will be placed on the counter."),
    ]
    assert items[0].target_coverage == .5
    assert not items[0].subject_targeted
    destination=candidate("destination", "The hand places the blue cutting board on the pink rectangular cutting board.")
    assert destination.target_coverage == 1.0 and not destination.subject_targeted
    comparisons=compare_events(items,entailment=FakeNLI(),threshold=.7)
    selected=select_unique_branches(items,comparisons=comparisons,branch_count=3,min_quality=.75,min_target_coverage=.75)
    assert [x.branch_id for x in selected] == ["move1", "place"]
    synonym=[candidate("lift","The hand lifts the cup.","cup"),candidate("take","A person picks up the mug.","mug")]
    assert compare_events(synonym,entailment=FakeNLI(),threshold=.7)[0].same_event
    prompt=build_feedback_prompt(question="What next?",target_object="pink rectangular cutting board",retained=selected,round_index=1)
    assert "must not be a paraphrase" in prompt and "Avoid 'or'" in prompt
    assert "beginning exactly" in prompt
    print("PASS: target guard, bidirectional semantic interface, variable unique selection, feedback prompt")


if __name__ == "__main__": main()
