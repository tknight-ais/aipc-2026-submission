"""Tournament: blind pairwise judging of the experiment's essays (step 5).

Essays are only compared with essays on the same thesis from a different
condition. Each essay meets about `per_essay` others, not all of them:
round-robin is too many pairs. Texts arrive already blinded
(blind.normalise).
"""

import random
from itertools import combinations

from pydantic import BaseModel

from aipc.context import Context
from aipc.prompt_loader import load_prompt
from aipc.stages import neutral_judge
from aipc.store import ExperimentEssay, Judgment


class Pair(BaseModel):
    thesis_id: str
    essay_a_run: str
    essay_b_run: str


class Schedule(BaseModel):
    experiment: str
    seed: int
    per_essay: int
    pairs: list[Pair]


def schedule_pairs(essays: list[ExperimentEssay], per_essay: int, seed: int) -> list[Pair]:
    """Seeded, greedy: shuffle every eligible pair, keep one while both
    essays still have fewer than `per_essay` opponents.
    """
    eligible = [
        (a, b) for a, b in combinations(essays, 2)
        if a.thesis_id == b.thesis_id and a.condition != b.condition
    ]
    random.Random(seed).shuffle(eligible)
    degree = {e.run_id: 0 for e in essays}
    pairs = []
    for a, b in eligible:
        if degree[a.run_id] < per_essay and degree[b.run_id] < per_essay:
            degree[a.run_id] += 1
            degree[b.run_id] += 1
            pairs.append(Pair(thesis_id=a.thesis_id, essay_a_run=a.run_id, essay_b_run=b.run_id))
    return pairs


def judge_ordering(pair: Pair, shown_first_run: str, texts: dict[str, str], judge_tier: str, seed: int, ctx: Context) -> tuple[str | None, str]:
    """Judges one ordering of a pair. Returns (winner run id or None, reasoning)."""
    first = shown_first_run
    second = pair.essay_b_run if first == pair.essay_a_run else pair.essay_a_run
    winner, reasoning = neutral_judge.compare(texts[first], texts[second], judge_tier, seed, ctx)
    winner_run = {"A": first, "B": second}.get(winner)
    return winner_run, reasoning


def run(experiment: str, schedule: Schedule, texts: dict[str, str], judge_tier: str, ctx: Context) -> list[Judgment]:
    """Judges every scheduled pair in both orderings, saving each verdict
    as it lands. Orderings this judge model already judged with this exact
    prompt are skipped, so a crashed tournament resumes. Nothing is saved
    on a dry run.
    """
    judge_model = ctx.config.model_for_tier(judge_tier).id
    _, prompt_hash = load_prompt(ctx.config.judge_prompt)
    done = {
        (j.essay_a_run, j.essay_b_run, j.shown_first_run)
        for j in ctx.store.judgments(experiment)
        if (j.judge_model, j.judge_prompt_hash) == (judge_model, prompt_hash)
    }
    judgments = []
    for pair in schedule.pairs:
        for shown_first in (pair.essay_a_run, pair.essay_b_run):
            if (pair.essay_a_run, pair.essay_b_run, shown_first) in done:
                continue
            winner_run, reasoning = judge_ordering(pair, shown_first, texts, judge_tier, ctx.config.defaults.seed, ctx)
            judgment = Judgment(
                experiment=experiment, tournament_run_id=ctx.run_id, thesis_id=pair.thesis_id,
                essay_a_run=pair.essay_a_run, essay_b_run=pair.essay_b_run, judge_model=judge_model,
                shown_first_run=shown_first, winner_run=winner_run, reasoning=reasoning,
                judge_prompt_hash=prompt_hash,
            )
            if not ctx.config.dry_run:
                ctx.store.save_judgment(judgment)
            judgments.append(judgment)
    return judgments
