"""A blind human ranking of a few essays, compared with the judge's order.

Rating 20 anchor pairs means reading 40 essays. Ranking one essay per
condition means reading each once. The essays are drawn without looking
at any verdict and shown under random codes, so the ranking is blind to
both condition and judge. The comparison uses each essay's own
Bradley-Terry strength from the judge's verdicts.
"""

import random
from itertools import combinations

from pydantic import BaseModel

from aipc.stages import analysis
from aipc.store import ExperimentEssay, Judgment

CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


class BlindEntry(BaseModel):
    code: str
    run_id: str
    condition: str


class BlindSet(BaseModel):
    experiment: str
    seed: int
    entries: list[BlindEntry]


class RankingAgreement(BaseModel):
    human_order: list[str]
    judge_order: list[str]
    concordant: int
    discordant: int
    tau: float


def draw_blind_set(experiment: str, essays: list[ExperimentEssay], seed: int) -> BlindSet:
    """One essay per condition, drawn with `seed`, each under a random
    two-letter code; the entries are listed in a shuffled order.
    """
    rng = random.Random(seed)
    by_condition: dict[str, list[ExperimentEssay]] = {}
    for essay in essays:
        by_condition.setdefault(essay.condition, []).append(essay)
    drawn = [rng.choice(by_condition[c]) for c in sorted(by_condition)]
    codes = rng.sample([a + b for a in CODE_LETTERS for b in CODE_LETTERS if a != b], len(drawn))
    entries = [BlindEntry(code=code, run_id=e.run_id, condition=e.condition) for code, e in zip(codes, drawn)]
    rng.shuffle(entries)
    return BlindSet(experiment=experiment, seed=seed, entries=entries)


def judge_order(judgments: list[Judgment], run_ids: list[str]) -> list[str]:
    """The essays in `run_ids`, best first, by Bradley-Terry strengths fitted
    to every essay in the tournament, not just these.
    """
    outcomes = analysis.pair_outcomes(judgments)
    every_essay = sorted({r for o in outcomes for r in (o.essay_a_run, o.essay_b_run)})
    strength = analysis.bradley_terry(outcomes, {r: r for r in every_essay}, every_essay)
    return sorted(run_ids, key=lambda r: -strength[r])


def agreement(human_order: list[str], judge_order: list[str]) -> RankingAgreement:
    """Kendall's tau between two orderings of the same essays."""
    position = {r: i for i, r in enumerate(judge_order)}
    concordant = sum(1 for a, b in combinations(human_order, 2) if position[a] < position[b])
    pairs = len(human_order) * (len(human_order) - 1) // 2
    discordant = pairs - concordant
    return RankingAgreement(
        human_order=human_order,
        judge_order=judge_order,
        concordant=concordant,
        discordant=discordant,
        tau=(concordant - discordant) / pairs,
    )
