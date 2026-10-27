"""The original pipeline (`aipc generate`), built before the component
ladder and kept because the project's earliest results came from it: a
pipeline essay judged against a fresh single-model baseline by the
persona panel. The current experiment is stages/experiment.py.

Assembles the stages into one end-to-end run: topic area in, a scored
essay out. Every stage is also exercised on its own by a single-stage CLI
command and its tests; this module is where they are chained.

Each stage's artifact is persisted as it completes (not batched at the
end), so a failure partway through a run still leaves everything prior
inspectable, and any stage can be replayed against fixed upstream input.

Known simplification: the pairwise reference essay is generated fresh in
every run rather than frozen once per topic and reused.
"""

from dataclasses import dataclass

from aipc.context import Context
from aipc.models import (
    ContentLedger,
    Decomposition,
    Essay,
    FidelityReview,
    RevisionRound,
    ScreenedThesis,
)
from aipc.stages import debate as debate_stage
from aipc.stages import decomposition as decomposition_stage
from aipc.stages import draft as draft_stage
from aipc.stages import ideation as ideation_stage
from aipc.stages import judge as judge_stage
from aipc.stages import literature as literature_stage
from aipc.stages.checks import MAX_WORDS


@dataclass
class PipelineResult:
    thesis: str
    decomposition: Decomposition
    ledger: ContentLedger
    essay: Essay
    rounds: list[RevisionRound]
    fidelity: FidelityReview
    reference_essay: Essay | None
    comparisons: dict[str, dict]


def _choose_thesis(screened: list[ScreenedThesis]) -> str:
    """Prefers a candidate that passed both filters; falls back to
    significance-only, then to the first candidate generated - ideation
    always returns at least one candidate, so this never has to fail.
    """
    survivors = [s for s in screened if s.survived]
    passed_significance = [s for s in screened if s.significance.passed]
    chosen_pool = survivors or passed_significance or screened
    return chosen_pool[0].candidate.claim


def write_baseline(topic: str, ctx: Context) -> Essay:
    """A single-model essay on the topic area: the pairwise reference.

    The model picks its own thesis from the bare topic and it isn't
    extracted, so thesis is left unknown rather than mislabeled.
    """
    word_min, word_max = ctx.config.limits.target_words
    result = ctx.ask("baseline", topic=topic, word_min=word_min, word_max=word_max)
    return Essay.from_text(result.text, thesis=None)


def run(topic: str, ctx: Context, evaluate: bool = True) -> PipelineResult:
    """Topic in, essay out. With `evaluate`, the persona panel then scores
    the essay against a fresh baseline; the essay does not depend on it.

    The ledger carries the topic and the literature stage's engagement list
    and sources to the writing stage, which otherwise sees none of them:
    without them the essay never learns what question it answers or which
    works to engage, and writes its references from memory.
    """
    lit_pack = literature_stage.run(topic, ctx)
    ctx.store.save_artifact(ctx.run_id, "literature", lit_pack)

    screened = ideation_stage.run(topic, lit_pack, ctx)
    for i, result in enumerate(screened):
        ctx.store.save_artifact(ctx.run_id, f"candidate{i}", result)
    thesis = _choose_thesis(screened)

    _, decomposition = decomposition_stage.run(thesis, ctx.config.pipeline.decomposition_candidates, ctx)
    ctx.store.save_artifact(ctx.run_id, "decomposition", decomposition)

    content_ledger, transcripts = debate_stage.run(decomposition, ctx)
    for i, transcript in enumerate(transcripts):
        ctx.store.save_artifact(ctx.run_id, f"transcript{i}", transcript)
    content_ledger = content_ledger.model_copy(
        update={"question": topic, "engagements": lit_pack.engagement_list, "sources": lit_pack.sources}
    )
    ctx.store.save_artifact(ctx.run_id, "ledger", content_ledger)

    essay, rounds, fidelity = draft_stage.run(content_ledger, ctx, max_words=MAX_WORDS)
    for revision_round in rounds:
        ctx.store.save_artifact(ctx.run_id, f"round{revision_round.round_num}", revision_round)
        ctx.store.save_measurement(ctx.run_id, "word_count", float(revision_round.word_count))
        ctx.store.save_measurement(ctx.run_id, "hedge_density", revision_round.hedge_density)
    ctx.store.save_artifact(ctx.run_id, "fidelity", fidelity)
    ctx.store.save_artifact(ctx.run_id, "essay", essay, markdown_text=essay.text)
    ctx.store.save_measurement(ctx.run_id, "word_count", float(essay.word_count))

    reference_essay, comparisons = _evaluate(essay, topic, ctx) if evaluate else (None, {})
    return PipelineResult(
        thesis=thesis,
        decomposition=decomposition,
        ledger=content_ledger,
        essay=essay,
        rounds=rounds,
        fidelity=fidelity,
        reference_essay=reference_essay,
        comparisons=comparisons,
    )


def _evaluate(essay: Essay, topic: str, ctx: Context) -> tuple[Essay, dict[str, dict]]:
    """The persona panel's absolute scores and pairwise verdicts against a
    fresh baseline essay on the same topic.
    """
    reference_essay = write_baseline(topic, ctx)
    ctx.store.save_artifact(ctx.run_id, "reference", reference_essay, markdown_text=reference_essay.text)

    comparisons: dict[str, dict] = {}
    for persona in judge_stage.MIRROR_PANEL:
        for score in judge_stage.score_absolute(
            essay, essay_id="candidate", persona=persona, panel_type="mirror", ctx=ctx
        ):
            ctx.store.save_score(ctx.run_id, score)

        comparison = judge_stage.compare_both_orderings(essay, reference_essay, persona, ctx)
        winner = "candidate" if comparison["candidate_wins"] else "reference"
        ctx.store.save_comparison("candidate", "reference", persona.name, winner, "both_orders")
        comparisons[persona.name] = comparison
    return reference_essay, comparisons
