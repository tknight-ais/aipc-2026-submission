"""Selection: which essay do we submit.

Every candidate meets every other, judged by the calibrated neutral judge
in both orderings for each seed, with formatting stripped as in the
tournament. A trial is a win only if both orderings pick the same essay;
otherwise it is a tie. Candidates are ranked by wins, then by wins minus
losses. Round-robin is affordable here because there are only a few
candidates; the tournament samples pairs instead.

This replaced an earlier selection by the persona panel, an uncalibrated
judge whose verdicts often flipped with the order of the essays.
"""

from itertools import combinations

from pydantic import BaseModel

from aipc.blind import normalise
from aipc.context import Context
from aipc.stages import neutral_judge


class Trial(BaseModel):
    essay_a: str
    essay_b: str
    seed: int
    winner_a_first: str | None
    winner_b_first: str | None
    winner: str | None


class EssayRanking(BaseModel):
    name: str
    wins: int
    losses: int
    trials: int


class SelectionResult(BaseModel):
    winner: str
    tied_at_top: bool
    ranking: list[EssayRanking]
    trials: list[Trial]


def judge_trial(name_a: str, name_b: str, texts: dict[str, str], judge_tier: str, seed: int, ctx: Context) -> Trial:
    first, _ = neutral_judge.compare(texts[name_a], texts[name_b], judge_tier, seed, ctx)
    second, _ = neutral_judge.compare(texts[name_b], texts[name_a], judge_tier, seed, ctx)
    winner_a_first = {"A": name_a, "B": name_b}[first] if first else None
    winner_b_first = {"A": name_b, "B": name_a}[second] if second else None
    winner = winner_a_first if winner_a_first == winner_b_first else None
    return Trial(
        essay_a=name_a,
        essay_b=name_b,
        seed=seed,
        winner_a_first=winner_a_first,
        winner_b_first=winner_b_first,
        winner=winner,
    )


def rank(names: list[str], trials: list[Trial]) -> list[EssayRanking]:
    rankings = []
    for name in names:
        mine = [t for t in trials if name in (t.essay_a, t.essay_b)]
        wins = sum(t.winner == name for t in mine)
        losses = sum(t.winner not in (None, name) for t in mine)
        rankings.append(EssayRanking(name=name, wins=wins, losses=losses, trials=len(mine)))
    return sorted(rankings, key=lambda r: (-r.wins, -(r.wins - r.losses)))


def run(candidates: dict[str, str], judge_tier: str, repeats: int, ctx: Context) -> SelectionResult:
    """`candidates` maps a name to essay text."""
    texts = {name: normalise(text) for name, text in candidates.items()}
    trials = [
        judge_trial(a, b, texts, judge_tier, ctx.config.defaults.seed + r, ctx)
        for a, b in combinations(texts, 2)
        for r in range(repeats)
    ]
    ranking = rank(list(texts), trials)
    top, *rest = ranking
    tied = bool(rest) and (rest[0].wins, rest[0].wins - rest[0].losses) == (top.wins, top.wins - top.losses)
    return SelectionResult(winner=top.name, tied_at_top=tied, ranking=ranking, trials=trials)
