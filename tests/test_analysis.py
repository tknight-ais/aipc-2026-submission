import math
import random

import pytest

from aipc.stages import analysis
from aipc.store import HumanJudgment, Judgment

CONDITIONS = ["A", "B", "C"]


def _judgment(a, b, shown_first, winner):
    return Judgment(
        experiment="x",
        tournament_run_id="t",
        thesis_id="t1",
        essay_a_run=a,
        essay_b_run=b,
        judge_model="m",
        shown_first_run=shown_first,
        winner_run=winner,
        reasoning="",
    )


def _outcome(a, b, winner):
    return analysis.PairOutcome(
        thesis_id="t1", essay_a_run=a, essay_b_run=b, winner_run=winner, position_consistent=winner is not None
    )


def test_pair_outcomes_needs_both_orderings_to_agree():
    judgments = [
        _judgment("a1", "b1", "a1", "a1"),
        _judgment("a1", "b1", "b1", "a1"),
        _judgment("a2", "b2", "a2", "a2"),
        _judgment("a2", "b2", "b2", "b2"),
        _judgment("a3", "b3", "a3", None),
        _judgment("a3", "b3", "b3", None),
    ]
    outcomes = {o.essay_a_run: o for o in analysis.pair_outcomes(judgments)}
    assert outcomes["a1"].winner_run == "a1" and outcomes["a1"].position_consistent
    assert outcomes["a2"].winner_run is None and not outcomes["a2"].position_consistent
    assert outcomes["a3"].winner_run is None and not outcomes["a3"].position_consistent


def test_pair_outcomes_refuses_a_pair_judged_in_only_one_ordering():
    judgments = [
        _judgment("a1", "b1", "a1", "a1"),
        _judgment("a1", "b1", "b1", "a1"),
        _judgment("a2", "b2", "a2", "a2"),
    ]
    with pytest.raises(ValueError, match="finish the tournament"):
        analysis.pair_outcomes(judgments)


def _simulated(true_log_strength, pairs_per_matchup, seed=0):
    """Outcomes drawn from a known Bradley-Terry model, one essay per condition per pair."""
    rng = random.Random(seed)
    outcomes, condition_of = [], {}
    n = 0
    for x in CONDITIONS:
        for y in CONDITIONS:
            if x >= y:
                continue
            p_x = 1 / (1 + math.exp(true_log_strength[y] - true_log_strength[x]))
            for _ in range(pairs_per_matchup):
                a, b = f"{x}{n}", f"{y}{n}"
                n += 1
                condition_of.update({a: x, b: y})
                outcomes.append(_outcome(a, b, a if rng.random() < p_x else b))
    return outcomes, condition_of


def test_bradley_terry_recovers_known_strengths():
    truth = {"A": 0.0, "B": 1.0, "C": -1.0}
    outcomes, condition_of = _simulated(truth, pairs_per_matchup=400)
    fitted = analysis.bradley_terry(outcomes, condition_of, CONDITIONS)
    assert fitted["A"] == 0.0
    assert fitted["B"] == pytest.approx(1.0, abs=0.2)
    assert fitted["C"] == pytest.approx(-1.0, abs=0.2)


def test_all_ties_give_equal_strengths_and_a_shutout_stays_finite():
    ties = [_outcome("a", "b", None), _outcome("a", "c", None)]
    condition_of = {"a": "A", "b": "B", "c": "C"}
    assert all(abs(v) < 1e-9 for v in analysis.bradley_terry(ties, condition_of, CONDITIONS).values())

    shutout = [_outcome("a", "b", "a"), _outcome("a", "c", "a")]
    fitted = analysis.bradley_terry(shutout, condition_of, CONDITIONS)
    assert all(math.isfinite(v) for v in fitted.values())
    assert fitted["B"] < 0 and fitted["C"] < 0


def test_bootstrap_interval_brackets_the_estimate():
    outcomes, condition_of = _simulated({"A": 0.0, "B": 1.0, "C": -1.0}, pairs_per_matchup=40)
    point = analysis.bradley_terry(outcomes, condition_of, CONDITIONS)
    intervals = analysis.bootstrap_ci(outcomes, condition_of, CONDITIONS)
    for c in ["B", "C"]:
        low, high = intervals[c]
        assert low <= point[c] <= high
    assert intervals["B"][0] > 0 > intervals["C"][1]


def test_essay_bootstrap_is_wider_when_essays_repeat_across_pairs():
    # Four essays, each in many pairs: 80 pairs but only 4 independent essays.
    condition_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    rng = random.Random(0)
    outcomes = [
        _outcome(a, b, a if rng.random() < 0.7 else b)
        for _ in range(20)
        for a, b in [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")]
    ]
    width = {
        name: analysis.bootstrap_ci(outcomes, condition_of, ["A", "B"], resample)["B"]
        for name, resample in [("essay", analysis.resample_essays), ("pair", analysis.resample_pairs)]
    }
    assert width["essay"][1] - width["essay"][0] > width["pair"][1] - width["pair"][0]


def test_win_rates_and_position_consistency():
    condition_of = {"a": "A", "b": "B", "c": "B"}
    outcomes = [_outcome("a", "b", "a"), _outcome("c", "a", "a"), _outcome("a", "c", None)]
    assert analysis.win_rates(outcomes, condition_of) == {("A", "B"): (2, 0, 1)}
    assert analysis.position_consistency(outcomes) == pytest.approx(2 / 3)


def test_verbosity_correlation_detects_longer_wins():
    words: dict[str, float] = {"a1": 100, "b1": 50, "a2": 40, "b2": 90, "a3": 120, "b3": 60}
    outcomes = [_outcome("a1", "b1", "a1"), _outcome("a2", "b2", "b2"), _outcome("a3", "b3", "a3")]
    correlation = analysis.verbosity_correlation(outcomes, words)
    assert correlation is not None and correlation > 0.9


def test_agreement_counts_judge_ties_as_disagreement():
    outcomes = [_outcome("a1", "b1", "a1"), _outcome("a2", "b2", "b2"), _outcome("a3", "b3", None)]
    human = [
        HumanJudgment(experiment="x", essay_a_run=a, essay_b_run=b, shown_first_run=a, winner_run=w, rated_at="t")
        for a, b, w in [("a1", "b1", "a1"), ("a2", "b2", "a2"), ("a3", "b3", "a3")]
    ]
    result = analysis.judge_human_agreement(human, outcomes)
    assert (result.rated, result.agreed) == (3, 1)


def test_binomial_tail_matches_the_preregistered_bar():
    assert analysis._binomial_upper_tail(14, 20) == pytest.approx(0.0577, abs=1e-4)
    assert analysis._binomial_upper_tail(0, 20) == 1.0


def test_longer_won_share_is_defined_when_correlation_is_not():
    words: dict[str, float] = {"a1": 10, "b1": 50, "a2": 10, "b2": 90, "a3": 10, "b3": 10}
    outcomes = [_outcome("a1", "b1", "b1"), _outcome("a2", "b2", "b2"), _outcome("a3", "b3", None)]
    assert analysis.verbosity_correlation(outcomes, words) is None
    assert analysis.longer_won_share(outcomes, words) == 1.0


def test_agreement_reports_a_clopper_pearson_interval():
    low, high = analysis._clopper_pearson(14, 20)
    assert (round(low, 3), round(high, 3)) == (0.457, 0.881)
    assert analysis._clopper_pearson(0, 5)[0] == 0.0
    assert analysis._clopper_pearson(5, 5)[1] == 1.0


def test_h4_gate_needs_the_whole_anchor_set_rated():
    human = [
        HumanJudgment(
            experiment="x",
            essay_a_run=f"a{i}",
            essay_b_run=f"b{i}",
            shown_first_run=f"a{i}",
            winner_run=f"a{i}",
            rated_at="t",
        )
        for i in range(20)
    ]
    judgments = [_judgment(f"a{i}", f"b{i}", s, f"a{i}") for i in range(20) for s in (f"a{i}", f"b{i}")]
    condition_of = {f"a{i}": "A" for i in range(20)} | {f"b{i}": "B" for i in range(20)}
    words = dict.fromkeys(condition_of, 100.0)
    args = ("x", "m", judgments)
    assert analysis.build_report(*args, human[:5], condition_of, words, words).h4_gate_pass is False
    assert analysis.build_report(*args, human, condition_of, words, words).h4_gate_pass is True


def test_factorial_effects_read_the_2x2():
    # Critique wins within each preparation level, and the debate room wins within each critique level.
    beats = [
        ("B_critique", "A_zero_shot"),
        ("D_debate", "D0_debate"),
        ("D0_debate", "A_zero_shot"),
        ("D_debate", "B_critique"),
        ("D_debate", "A_zero_shot"),
    ]
    judgments = [
        j
        for winner, loser in beats
        for j in (_judgment(winner, loser, winner, winner), _judgment(winner, loser, loser, winner))
    ]
    condition_of = {c: c for c in analysis.FACTORIAL_CELLS}
    outcomes = analysis.pair_outcomes(judgments)

    effects = analysis.factorial_effects(outcomes, condition_of, list(analysis.FACTORIAL_CELLS))

    assert effects is not None
    assert effects["critique"].estimate > 0 and effects["debate_room"].estimate > 0
    assert effects["critique"].ci_low <= effects["critique"].estimate <= effects["critique"].ci_high
    assert analysis.factorial_effects(outcomes, condition_of, ["A_zero_shot", "B_critique"]) is None
