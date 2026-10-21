"""Judge calibration test: gates all comparison work.

Before the judge panel is trusted to compare variants, this constructs a
known ordering - an essay, deliberately and cumulatively damaged at
graded levels - and checks whether the panel's absolute scores and
pairwise comparisons reproduce it. If the panel can't detect *constructed*
damage, it can't be trusted to detect the real quality differences this
project exists to measure.

The undamaged anchor is one of this project's own generated essays, not
a published paper. A published paper is the more realistic anchor, but a
generated essay sidesteps training-data memorization (a model has never
seen this exact essay before), and is available
without a literature-retrieval dependency. Worth revisiting with a real
published paper once one is on hand.
"""

from pydantic import BaseModel

from aipc.context import Context
from aipc.models import Essay
from aipc.stages import judge as judge_stage

# Subtly introducing a fallacy while keeping prose fluent is a harder task
# than the cheap tier should be trusted with.
GENERATION_TIER = "standard"

# Cumulative, worst-case-ordered: each level's damage is applied on top
# of the previous level's output, so "heavy" strictly contains "light"'s
# damage plus more - the ordering original > light > heavy > severe is
# true by construction, not just intended.
DAMAGE_LEVELS = [
    (
        "light",
        "Weaken exactly one objection's reply so it no longer actually "
        "answers the objection it responds to - make the reply sound "
        "confident but subtly miss the point. Leave every other claim, "
        "objection and reply as strong as in the original.",
    ),
    (
        "heavy",
        "Remove the support for the essay's central claim (state the "
        "claim but drop or hollow out the reasoning behind it), and "
        "introduce one clear logical fallacy in a key inference (for "
        "example affirming the consequent, or a false dichotomy) "
        "elsewhere in the argument.",
    ),
    (
        "severe",
        "Strip out nearly all objections and replies - keep only the "
        "bare thesis statement and generic, surface-level support for "
        "it, the way a rushed one-shot essay with no real "
        "argumentation would read.",
    ),
]


class CalibrationResult(BaseModel):
    persona: str
    absolute_scores: dict[str, float]
    pairwise_results: dict[str, bool]
    monotonic: bool


def degrade(essay: Essay, damage_instruction: str, ctx: Context) -> Essay:
    result = ctx.ask("degrade_essay", tier=GENERATION_TIER, essay_text=essay.text, damage_instruction=damage_instruction)
    return Essay.from_text(result.text, thesis=essay.thesis)


def build_graded_set(original: Essay, ctx: Context) -> list[tuple[str, Essay]]:
    levels = [("original", original)]
    current = original
    for name, instruction in DAMAGE_LEVELS:
        current = degrade(current, instruction, ctx)
        levels.append((name, current))
    return levels


def score_levels(levels: list[tuple[str, Essay]], ctx: Context) -> list[CalibrationResult]:
    results = []
    for persona in judge_stage.MIRROR_PANEL:
        absolute_scores = {}
        for name, essay in levels:
            scores = judge_stage.score_absolute(essay, essay_id=name, persona=persona, panel_type="calibration", ctx=ctx)
            absolute_scores[name] = sum(s.score for s in scores) / len(scores) if scores else 0.0

        pairwise_results = {}
        for (name_a, essay_a), (name_b, essay_b) in zip(levels, levels[1:]):
            comparison = judge_stage.compare_both_orderings(essay_a, essay_b, persona, ctx)
            pairwise_results[f"{name_a}_vs_{name_b}"] = comparison["candidate_wins"]

        level_names = [name for name, _ in levels]
        monotonic = all(
            absolute_scores[level_names[i]] >= absolute_scores[level_names[i + 1]]
            for i in range(len(level_names) - 1)
        )

        results.append(CalibrationResult(
            persona=persona.name,
            absolute_scores=absolute_scores,
            pairwise_results=pairwise_results,
            monotonic=monotonic,
        ))
    return results


def run(original: Essay, ctx: Context) -> tuple[list[tuple[str, Essay]], list[CalibrationResult]]:
    levels = build_graded_set(original, ctx)
    return levels, score_levels(levels, ctx)
