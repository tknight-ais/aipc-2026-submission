"""Component ladder, conditions A-D: each step adds one thing.

All conditions write to the same fixed thesis with the same generator,
and B-D get exactly one critique round so the critique dose is constant:

- A_zero_shot: one call, thesis in, essay out.
- B_critique: A, then one round of parallel critics and one revision.
- C_structured: decomposition, flat ledger, draft, one critique round.
- D_debate: C with the debate room in place of the flat ledger.
- E_frontier_zero_shot: A, but written by the frontier tier. Not a ladder
  step: it asks whether a scaffolded cheap model beats a plain strong one.
"""

from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel

from aipc.config import Config
from aipc.context import Context
from aipc.models import Essay
from aipc.stages import debate as debate_stage
from aipc.stages import decomposition as decomposition_stage
from aipc.stages import draft as draft_stage

FRONTIER_TIER = "frontier"


class Thesis(BaseModel):
    id: str
    claim: str


class ExperimentConfig(BaseModel):
    """`target_words`, when set, replaces `limits.target_words` for every
    essay in the experiment, so an experiment's length doesn't move when the
    default does.
    """

    name: str
    seed: int
    essays_per_condition: int
    generator_tier: str
    conditions: list[str]
    theses: list[Thesis]
    target_words: tuple[int, int] | None = None


def load_experiment(path: Path) -> ExperimentConfig:
    experiment = ExperimentConfig(**yaml.safe_load(path.read_text()))
    unknown = [c for c in experiment.conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {', '.join(unknown)}")
    if not experiment.theses:
        raise ValueError(f"{path} lists no theses; draw them first (ladder step 1)")
    return experiment


def condition_config(
    base: Config, condition: str, generator_tier: str, seed: int, target_words: tuple[int, int] | None = None
) -> Config:
    """The resolved config for one essay: one critique round, the
    condition's ledger strategy and generator tier, the essay's own seed,
    and the experiment's word target if it sets one.
    """
    strategy = "debate_room" if condition == "D_debate" else "flat"
    tier = FRONTIER_TIER if condition == "E_frontier_zero_shot" else generator_tier
    pipeline = base.pipeline.model_copy(update={"strategy": strategy, "max_revision_rounds": 1})
    defaults = base.defaults.model_copy(update={"seed": seed, "tier": tier})
    limits = base.limits.model_copy(update={"target_words": target_words}) if target_words else base.limits
    return base.model_copy(update={"pipeline": pipeline, "defaults": defaults, "limits": limits})


def write_zero_shot(thesis: str, ctx: Context) -> Essay:
    result = ctx.ask("zero_shot_thesis", thesis=thesis, requirements=draft_stage.essay_requirements(ctx))
    return Essay.from_text(result.text, thesis=thesis)


def write_critique(thesis: str, ctx: Context) -> Essay:
    draft = write_zero_shot(thesis, ctx)
    ctx.store.save_artifact(ctx.run_id, "draft", draft, markdown_text=draft.text)

    record, objections = draft_stage.critique_round(draft, thesis, [], round_num=1, ctx=ctx)
    ctx.store.save_artifact(ctx.run_id, "round1", record)
    if not objections:
        return draft
    essay = draft_stage.revise_free(draft, objections, ctx)
    essay.rounds_used = 1
    return essay


def write_structured(thesis: str, ctx: Context) -> Essay:
    """C and D; the ledger strategy comes from the condition's config."""
    _, decomposition = decomposition_stage.run(thesis, ctx.config.pipeline.decomposition_candidates, ctx)
    ctx.store.save_artifact(ctx.run_id, "decomposition", decomposition)

    ledger, transcripts = debate_stage.run(decomposition, ctx)
    for i, transcript in enumerate(transcripts):
        ctx.store.save_artifact(ctx.run_id, f"transcript{i}", transcript)
    ctx.store.save_artifact(ctx.run_id, "ledger", ledger)

    essay, rounds, fidelity = draft_stage.run(ledger, ctx)
    for revision_round in rounds:
        ctx.store.save_artifact(ctx.run_id, f"round{revision_round.round_num}", revision_round)
    ctx.store.save_artifact(ctx.run_id, "fidelity", fidelity)
    return essay


CONDITIONS: dict[str, Callable[[str, Context], Essay]] = {
    "A_zero_shot": write_zero_shot,
    "B_critique": write_critique,
    "C_structured": write_structured,
    "D_debate": write_structured,
    "E_frontier_zero_shot": write_zero_shot,
}


def write_essay(condition: str, thesis: str, ctx: Context) -> tuple[Essay, Path]:
    """Writes one essay and persists it with its word count and hedge
    density. Returns the essay and the path it was saved to.
    """
    essay = CONDITIONS[condition](thesis, ctx)
    path = ctx.store.save_artifact(ctx.run_id, "essay", essay, markdown_text=essay.text)
    ctx.store.save_measurement(ctx.run_id, "word_count", float(essay.word_count))
    ctx.store.save_measurement(ctx.run_id, "hedge_density", draft_stage.hedge_density(essay.text))
    return essay, path
