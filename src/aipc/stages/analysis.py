"""Component-ladder analysis: the pre-registered primary and secondary
analyses. Standard library only.

A pair's outcome is its winner only if both orderings agree; otherwise it
is a tie, as pre-registered. The pre-registered interval resamples pairs;
an essay-resampled interval is added as a robustness check, because pairs
that share an essay are not independent. Bradley-Terry counts a tie as half a win to
each side, and every pair of conditions gets one virtual tie, so a
condition that never wins still has a finite strength. That prior is not
in the pre-registration; it only matters when a condition wins nothing.
"""

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations

from pydantic import BaseModel

from aipc.store import HumanJudgment, Judgment

BT_ITERATIONS = 500
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 1
VIRTUAL_TIES = 1.0
POSITION_CONSISTENCY_BAR = 0.7
AGREEMENT_BAR = 14 / 20
# The gate needs the whole anchor set rated, not a lucky first few.
AGREEMENT_MIN_RATED = 20


class PairOutcome(BaseModel):
    thesis_id: str
    essay_a_run: str
    essay_b_run: str
    winner_run: str | None
    position_consistent: bool


def pair_outcomes(judgments: list[Judgment]) -> list[PairOutcome]:
    """Collapses each pair's two orderings into one outcome."""
    by_pair = defaultdict(list)
    for j in judgments:
        by_pair[(j.thesis_id, j.essay_a_run, j.essay_b_run)].append(j.winner_run)
    outcomes = []
    for (thesis_id, a, b), winners in sorted(by_pair.items()):
        consistent = len(winners) == 2 and winners[0] is not None and winners[0] == winners[1]
        outcomes.append(PairOutcome(
            thesis_id=thesis_id, essay_a_run=a, essay_b_run=b,
            winner_run=winners[0] if consistent else None, position_consistent=consistent,
        ))
    return outcomes


def _wins(outcomes: list[PairOutcome], condition_of: dict[str, str]) -> Counter:
    """wins[(x, y)] = how often condition x beat condition y; ties count half."""
    wins = Counter()
    for o in outcomes:
        a, b = condition_of[o.essay_a_run], condition_of[o.essay_b_run]
        if o.winner_run is None:
            wins[(a, b)] += 0.5
            wins[(b, a)] += 0.5
        else:
            loser = b if o.winner_run == o.essay_a_run else a
            wins[(condition_of[o.winner_run], loser)] += 1
    return wins


def bradley_terry(outcomes: list[PairOutcome], condition_of: dict[str, str], conditions: list[str]) -> dict[str, float]:
    """Log-strength per condition, relative to the first condition (Hunter's MM)."""
    wins = _wins(outcomes, condition_of)
    for x, y in combinations(conditions, 2):
        wins[(x, y)] += VIRTUAL_TIES / 2
        wins[(y, x)] += VIRTUAL_TIES / 2
    strength = {c: 1.0 for c in conditions}
    for _ in range(BT_ITERATIONS):
        for i in conditions:
            total_wins = sum(wins[(i, j)] for j in conditions if j != i)
            denominator = sum(
                (wins[(i, j)] + wins[(j, i)]) / (strength[i] + strength[j]) for j in conditions if j != i
            )
            strength[i] = total_wins / denominator
    reference = strength[conditions[0]]
    return {c: math.log(strength[c] / reference) for c in conditions}


def resample_pairs(outcomes: list[PairOutcome], rng: random.Random) -> list[PairOutcome]:
    """Pairs drawn with replacement, as pre-registered. Treats pairs as
    independent, which they are not - each essay sits in several pairs - so
    the interval is too narrow when essays differ in quality.
    """
    return rng.choices(outcomes, k=len(outcomes))


def resample_essays(outcomes: list[PairOutcome], rng: random.Random) -> list[PairOutcome]:
    """Essays drawn with replacement; a pair is kept once per combination of
    its essays' draws, so pairs sharing an essay move together. This is the
    vertex bootstrap for dyadic data (Snijders & Borgatti 1999, Connections
    22(2)), which errs wide: it overstates pair-level noise, a lot when
    essays are rarely shared. (Not Owen's pigeonhole bootstrap, which is
    for two crossed factors: here one set of essays sits on both sides of
    a pair.)

    Known flaw: with few essays per condition a resample can drop every
    essay of a condition, leaving that condition's strength to the
    virtual-tie prior, which widens the tails. Not corrected.
    """
    essays = sorted({run for o in outcomes for run in (o.essay_a_run, o.essay_b_run)})
    drawn = Counter(rng.choices(essays, k=len(essays)))
    return [o for o in outcomes for _ in range(drawn[o.essay_a_run] * drawn[o.essay_b_run])]


def bootstrap_ci(
    outcomes: list[PairOutcome], condition_of: dict[str, str], conditions: list[str],
    resample: Callable[[list[PairOutcome], random.Random], list[PairOutcome]] = resample_pairs,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, tuple[float, float]]:
    """95% percentile interval for each log-strength."""
    rng = random.Random(BOOTSTRAP_SEED)
    samples = defaultdict(list)
    for _ in range(resamples):
        for c, value in bradley_terry(resample(outcomes, rng), condition_of, conditions).items():
            samples[c].append(value)
    return {c: _percentiles(samples[c]) for c in conditions}


def _percentiles(values: list[float]) -> tuple[float, float]:
    cuts = statistics.quantiles(values, n=40, method="inclusive")
    return cuts[0], cuts[-1]


def win_rates(outcomes: list[PairOutcome], condition_of: dict[str, str]) -> dict[tuple[str, str], tuple[int, int, int]]:
    """For each condition pair (x, y), x before y alphabetically: (x wins, y wins, ties)."""
    table = defaultdict(lambda: [0, 0, 0])
    for o in outcomes:
        x, y = sorted((condition_of[o.essay_a_run], condition_of[o.essay_b_run]))
        if o.winner_run is None:
            table[(x, y)][2] += 1
        else:
            table[(x, y)][0 if condition_of[o.winner_run] == x else 1] += 1
    return {k: tuple(v) for k, v in sorted(table.items())}


def position_consistency(outcomes: list[PairOutcome]) -> float:
    return sum(o.position_consistent for o in outcomes) / len(outcomes)


def verbosity_correlation(outcomes: list[PairOutcome], word_count: dict[str, float]) -> float | None:
    """Correlation between 'A won' (+1/-1) and A's word-count lead, over decided pairs."""
    decided = [o for o in outcomes if o.winner_run is not None]
    if len(decided) < 3:
        return None
    won = [1.0 if o.winner_run == o.essay_a_run else -1.0 for o in decided]
    lead = [word_count[o.essay_a_run] - word_count[o.essay_b_run] for o in decided]
    if len(set(won)) < 2 or len(set(lead)) < 2:
        return None
    return statistics.correlation(won, lead)


def longer_won_share(outcomes: list[PairOutcome], word_count: dict[str, float]) -> float | None:
    """Share of decided pairs won by the longer essay. Unlike the
    correlation, defined even when one condition wins every pair.
    """
    decided = [o for o in outcomes if o.winner_run is not None and word_count[o.essay_a_run] != word_count[o.essay_b_run]]
    if not decided:
        return None
    longer = [max((o.essay_a_run, o.essay_b_run), key=word_count.__getitem__) for o in decided]
    return sum(o.winner_run == run for o, run in zip(decided, longer)) / len(decided)


def mean_by_condition(values: dict[str, float], condition_of: dict[str, str]) -> dict[str, float]:
    grouped = defaultdict(list)
    for run_id, value in values.items():
        grouped[condition_of[run_id]].append(value)
    return {c: statistics.mean(v) for c, v in sorted(grouped.items())}


class Agreement(BaseModel):
    """ci_* is the 95% Clopper-Pearson interval on the agreement rate:
    with 20 anchors, 14 agreements is anywhere from 46% to 88%.
    """

    rated: int
    agreed: int
    kappa: float | None
    binomial_p: float
    ci_low: float | None
    ci_high: float | None


def judge_human_agreement(human: list[HumanJudgment], outcomes: list[PairOutcome]) -> Agreement:
    """A judge tie never agrees with the human rater, who must pick one.
    Anchors are drawn from decided pairs only (see viewer/rating.py).
    """
    judge_winner = {(o.essay_a_run, o.essay_b_run): o.winner_run for o in outcomes}
    labels = [(_side(h, h.winner_run), _side(h, judge_winner[(h.essay_a_run, h.essay_b_run)])) for h in human]
    agreed = sum(a == b for a, b in labels)
    ci = _clopper_pearson(agreed, len(labels)) if labels else (None, None)
    return Agreement(
        rated=len(labels), agreed=agreed, kappa=_kappa(labels),
        binomial_p=_binomial_upper_tail(agreed, len(labels)), ci_low=ci[0], ci_high=ci[1],
    )


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(predicate: Callable[[float], bool]) -> float:
    """Smallest p in [0, 1] where a monotone predicate turns true."""
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (low, mid) if predicate(mid) else (mid, high)
    return high


def _clopper_pearson(k: int, n: int) -> tuple[float, float]:
    low = 0.0 if k == 0 else _bisect(lambda p: 1 - _binomial_cdf(k - 1, n, p) >= 0.025)
    high = 1.0 if k == n else _bisect(lambda p: _binomial_cdf(k, n, p) <= 0.025)
    return low, high


def _side(h: HumanJudgment, winner_run: str | None) -> str:
    return "tie" if winner_run is None else ("a" if winner_run == h.essay_a_run else "b")


def _kappa(labels: list[tuple[str, str]]) -> float | None:
    if not labels:
        return None
    n = len(labels)
    observed = sum(a == b for a, b in labels) / n
    first, second = Counter(a for a, _ in labels), Counter(b for _, b in labels)
    expected = sum(first[k] * second[k] for k in first) / n**2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _binomial_upper_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5): agreement this high by chance."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


class StrengthEstimate(BaseModel):
    """ci_* resamples pairs (pre-registered; errs narrow). essay_ci_*
    resamples essays (robustness check; errs wide). A difference that
    clears both is robust to how pairs share essays.
    """

    log_strength: float
    ci_low: float
    ci_high: float
    essay_ci_low: float
    essay_ci_high: float


class Report(BaseModel):
    experiment: str
    judge_model: str
    essays: int
    pairs: int
    strengths: dict[str, dict[str, StrengthEstimate]]
    win_rates: list[str]
    position_consistency: float
    verbosity_correlation: float | None
    longer_won_share: float | None
    hedge_density: dict[str, float]
    word_count: dict[str, float]
    agreement: Agreement
    position_consistency_pass: bool
    h4_gate_pass: bool


def _strengths(outcomes: list[PairOutcome], condition_of: dict[str, str], conditions: list[str]) -> dict[str, StrengthEstimate]:
    point = bradley_terry(outcomes, condition_of, conditions)
    by_essay = bootstrap_ci(outcomes, condition_of, conditions, resample_essays)
    by_pair = bootstrap_ci(outcomes, condition_of, conditions, resample_pairs)
    return {
        c: StrengthEstimate(
            log_strength=point[c], ci_low=by_pair[c][0], ci_high=by_pair[c][1],
            essay_ci_low=by_essay[c][0], essay_ci_high=by_essay[c][1],
        )
        for c in conditions
    }


def build_report(
    experiment: str, judge_model: str, judgments: list[Judgment], human: list[HumanJudgment],
    condition_of: dict[str, str], word_count: dict[str, float], hedge: dict[str, float],
) -> Report:
    """Primary analysis per thesis and pooled, every secondary analysis, and both gates."""
    outcomes = pair_outcomes(judgments)
    conditions = sorted(set(condition_of.values()))
    strengths = {"pooled": _strengths(outcomes, condition_of, conditions)}
    for thesis_id in sorted({o.thesis_id for o in outcomes}):
        strengths[thesis_id] = _strengths([o for o in outcomes if o.thesis_id == thesis_id], condition_of, conditions)
    agreement = judge_human_agreement(human, outcomes)
    consistency = position_consistency(outcomes)
    return Report(
        experiment=experiment,
        judge_model=judge_model,
        essays=len(condition_of),
        pairs=len(outcomes),
        strengths=strengths,
        win_rates=[f"{x} vs {y}: {w[0]}-{w[1]}, {w[2]} ties" for (x, y), w in win_rates(outcomes, condition_of).items()],
        position_consistency=consistency,
        verbosity_correlation=verbosity_correlation(outcomes, word_count),
        longer_won_share=longer_won_share(outcomes, word_count),
        hedge_density=mean_by_condition(hedge, condition_of),
        word_count=mean_by_condition(word_count, condition_of),
        agreement=agreement,
        position_consistency_pass=consistency >= POSITION_CONSISTENCY_BAR,
        h4_gate_pass=agreement.rated >= AGREEMENT_MIN_RATED and agreement.agreed / agreement.rated >= AGREEMENT_BAR,
    )
