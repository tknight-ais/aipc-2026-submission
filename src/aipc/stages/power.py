"""Simulated power for a component-ladder design: how often the
pre-registered analysis would detect a true difference of a given size.

Model: an essay's quality is its condition's log-strength plus
N(0, essay_sd). Each scheduled pair is a judge tie with probability
tie_rate (the share of pairs whose orderings disagree); otherwise its
winner is drawn Bradley-Terry style from the quality difference. Only the
second condition differs from the first. The analysis is the real one:
the same schedule, Bradley-Terry fit and both bootstrap intervals.
"""

import math
import random

from pydantic import BaseModel

from aipc.stages import analysis
from aipc.stages.tournament import Pair, schedule_pairs
from aipc.store import ExperimentEssay


class PowerResult(BaseModel):
    """pair_power: the pre-registered pair interval excludes zero.
    robust_power: both the pair and the essay interval exclude zero.
    """

    essays_per_condition: int
    effect: float
    essay_sd: float
    judgments: int
    pair_power: float
    robust_power: float
    reps: int


def _outcome(pair: Pair, quality: dict[str, float], tie_rate: float, rng: random.Random) -> analysis.PairOutcome:
    a, b = pair.essay_a_run, pair.essay_b_run
    if rng.random() < tie_rate:
        winner = None
    else:
        a_wins = rng.random() < 1 / (1 + math.exp(quality[b] - quality[a]))
        winner = a if a_wins else b
    return analysis.PairOutcome(
        thesis_id=pair.thesis_id,
        essay_a_run=a,
        essay_b_run=b,
        winner_run=winner,
        position_consistent=winner is not None,
    )


def _excludes_zero(interval: tuple[float, float]) -> bool:
    return interval[0] > 0 or interval[1] < 0


def simulate(
    conditions: list[str],
    theses: int,
    essays_per_condition: int,
    per_essay: int,
    effect: float,
    essay_sd: float,
    tie_rate: float,
    reps: int,
    resamples: int,
    seed: int,
) -> PowerResult:
    rng = random.Random(seed)
    target = conditions[1]
    essays = [
        ExperimentEssay(thesis_id=f"t{t}", condition=c, essay_index=i, run_id=f"t{t}-{c}-{i}", essay_path="")
        for t in range(theses)
        for c in conditions
        for i in range(essays_per_condition)
    ]
    condition_of = {e.run_id: e.condition for e in essays}
    pair_hits = robust_hits = judgments = 0
    for _ in range(reps):
        quality = {e.run_id: (effect if e.condition == target else 0.0) + rng.gauss(0, essay_sd) for e in essays}
        pairs = schedule_pairs(essays, per_essay, seed=rng.randrange(2**31))
        outcomes = [_outcome(p, quality, tie_rate, rng) for p in pairs]
        by_pair = analysis.bootstrap_ci(outcomes, condition_of, conditions, analysis.resample_pairs, resamples)[target]
        by_essay = analysis.bootstrap_ci(outcomes, condition_of, conditions, analysis.resample_essays, resamples)[
            target
        ]
        pair_hits += _excludes_zero(by_pair)
        robust_hits += _excludes_zero(by_pair) and _excludes_zero(by_essay)
        judgments = 2 * len(pairs)
    return PowerResult(
        essays_per_condition=essays_per_condition,
        effect=effect,
        essay_sd=essay_sd,
        judgments=judgments,
        pair_power=pair_hits / reps,
        robust_power=robust_hits / reps,
        reps=reps,
    )
