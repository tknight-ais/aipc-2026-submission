from aipc.stages import ranking
from aipc.store import ExperimentEssay, Judgment


def _essay(condition, index):
    return ExperimentEssay(
        thesis_id="t1", condition=condition, essay_index=index, run_id=f"{condition}{index}", essay_path="x"
    )


def _both_orders(a, b, winner):
    """A pair judged in both orders, the same winner each time."""
    return [
        Judgment(
            experiment="x",
            tournament_run_id="t",
            thesis_id="t1",
            essay_a_run=a,
            essay_b_run=b,
            judge_model="m",
            shown_first_run=first,
            winner_run=winner,
            reasoning="r",
        )
        for first in (a, b)
    ]


def test_the_blind_set_has_one_essay_per_condition_under_distinct_codes():
    essays = [_essay(c, i) for c in ("A", "B", "D") for i in range(3)]
    drawn = ranking.draw_blind_set("x", essays, seed=1)

    assert sorted(e.condition for e in drawn.entries) == ["A", "B", "D"]
    assert len({e.code for e in drawn.entries}) == 3
    assert ranking.draw_blind_set("x", essays, seed=1) == drawn


def test_judge_order_follows_the_verdicts():
    judgments = _both_orders("A0", "B0", "B0") + _both_orders("B0", "D0", "D0") + _both_orders("A0", "D0", "D0")
    assert ranking.judge_order(judgments, ["A0", "B0", "D0"]) == ["D0", "B0", "A0"]


def test_agreement_counts_pairs_in_the_same_order():
    same = ranking.agreement(["x", "y", "z"], ["x", "y", "z"])
    reversed_ = ranking.agreement(["x", "y", "z"], ["z", "y", "x"])
    one_swap = ranking.agreement(["x", "z", "y"], ["x", "y", "z"])

    assert (same.concordant, same.tau) == (3, 1.0)
    assert (reversed_.concordant, reversed_.tau) == (0, -1.0)
    assert (one_swap.concordant, one_swap.discordant) == (2, 1)
