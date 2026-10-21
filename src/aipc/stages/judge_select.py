"""Judge selection: which candidate model judges the component ladder.

Each candidate judges the saved graded-damage essays from an earlier
`aipc calibrate` run (original > light > heavy > severe, true by
construction). For every adjacent pair it judges both orderings,
repeated with different seeds. A trial counts as correct only if the
less-damaged essay wins in both orderings.

Ranking: most correct trials, then most position-consistent trials, then
lowest cost. Fixed here, before any live run, so the pick is automatic.
"""

from pathlib import Path

from pydantic import BaseModel

from aipc.context import Context
from aipc.models import Essay
from aipc.stages import neutral_judge

LEVEL_ORDER = ["original", "light", "heavy", "severe"]


class Verdict(BaseModel):
    pair: str
    repeat: int
    seed: int
    better_position: str
    winner: str | None
    reasoning: str


class JudgeSelectionResult(BaseModel):
    judge_tier: str
    model: str
    trials: int
    correct: int
    position_consistent: int
    unparsed: int
    cost_usd: float
    verdicts: list[Verdict]


def load_levels(run_dir: Path) -> list[tuple[str, Essay]]:
    return [
        (name, Essay.model_validate_json((run_dir / f"level_{name}.json").read_text()))
        for name in LEVEL_ORDER
    ]


def _judge_trial(pair: str, better: Essay, worse: Essay, judge_tier: str, repeat: int, seed: int, ctx: Context) -> list[Verdict]:
    verdicts = []
    for better_position, (essay_a, essay_b) in [("A", (better, worse)), ("B", (worse, better))]:
        winner, reasoning = neutral_judge.compare(essay_a.text, essay_b.text, judge_tier, seed, ctx)
        verdicts.append(Verdict(
            pair=pair, repeat=repeat, seed=seed,
            better_position=better_position, winner=winner, reasoning=reasoning,
        ))
    return verdicts


def summarise(judge_tier: str, model: str, verdicts: list[Verdict], cost_usd: float) -> JudgeSelectionResult:
    """Scores verdicts in (better-as-A, better-as-B) pairs, as _judge_trial emits them."""
    trials = list(zip(verdicts[::2], verdicts[1::2]))
    picked_better = [[v.winner == v.better_position for v in trial] for trial in trials]
    parsed = [all(v.winner is not None for v in trial) for trial in trials]
    return JudgeSelectionResult(
        judge_tier=judge_tier,
        model=model,
        trials=len(trials),
        correct=sum(all(p) for p in picked_better),
        position_consistent=sum(ok and p[0] == p[1] for ok, p in zip(parsed, picked_better)),
        unparsed=sum(v.winner is None for v in verdicts),
        cost_usd=cost_usd,
        verdicts=verdicts,
    )


def run_judge(levels: list[tuple[str, Essay]], judge_tier: str, repeats: int, ctx: Context) -> JudgeSelectionResult:
    cost_before = ctx.llm.total_cost_usd
    verdicts = []
    for (better_name, better), (worse_name, worse) in zip(levels, levels[1:]):
        for repeat in range(repeats):
            seed = ctx.config.defaults.seed + repeat
            verdicts += _judge_trial(f"{better_name}_vs_{worse_name}", better, worse, judge_tier, repeat, seed, ctx)
    model = ctx.config.model_for_tier(judge_tier).id
    return summarise(judge_tier, model, verdicts, ctx.llm.total_cost_usd - cost_before)


def rank(results: list[JudgeSelectionResult]) -> list[JudgeSelectionResult]:
    return sorted(results, key=lambda r: (-r.correct, -r.position_consistent, r.cost_usd))
