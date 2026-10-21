"""Judge panel: persona judges score essays independently - never
conferring - using absolute rubric scores for selection and pairwise
comparison against a frozen per-topic reference for cross-variant
comparison, kept separate because selection and measurement are
different jobs.

MIRROR_PANEL is the frozen comparison instrument: changing it means
re-running baseline configurations through the new version, or
cross-variant comparisons stop meaning anything. Topic specialists are
swapped per topic and passed in by the caller, not hardcoded here.

model_tier is "standard" because cheap-tier judges failed the
graded-damage calibration test (calibrate.py), missing the subtle damage
that standard-tier judges caught.

The component ladder does not use this panel; it uses neutral_judge.py.
"""

import re

from aipc.context import Context
from aipc.models import Essay, Persona, Scorecard
from aipc.replies import parse_blocks, parse_pairwise

CRITERIA = ["argument_quality", "rigor", "originality", "clarity"]
SCORE_KEYS = {"CRITERION", "SCORE", "REASONING"}
SCORE_NUMBER = re.compile(r"\d+(?:\.\d+)?")

MIRROR_PANEL = [
    Persona(
        name="formal_epistemologist",
        model_tier="standard",
        description=(
            "a formal epistemologist, focused on precise argument structure, "
            "inference validity, and whether informal claims could be made rigorous"
        ),
    ),
    Persona(
        name="philosopher_of_mind",
        model_tier="standard",
        description=(
            "a philosopher of mind, focused on conceptual clarity about mental/physical "
            "categories and whether the essay's distinctions survive standard objections"
        ),
    ),
    Persona(
        name="philosopher_of_language",
        model_tier="standard",
        description=(
            "a philosopher of language, focused on whether key terms are used "
            "consistently and whether the argument trades on ambiguity"
        ),
    ),
    Persona(
        name="ethicist",
        model_tier="standard",
        description=(
            "an ethicist and axiologist, focused on whether normative claims are "
            "adequately motivated and whether the strongest opposing values-based "
            "objections are engaged"
        ),
    ),
]


def _score(block: dict[str, str]) -> float | None:
    match = SCORE_NUMBER.fullmatch(block.get("SCORE", "").strip())
    return float(match.group()) if match else None


def score_absolute(essay: Essay, essay_id: str, persona: Persona, panel_type: str, ctx: Context) -> list[Scorecard]:
    """One Scorecard per rubric block with a readable score; a block whose
    score can't be read is dropped rather than scored 0.
    """
    result = ctx.ask("judge_absolute", tier=persona.model_tier, persona_description=persona.description, essay_text=essay.text)
    blocks = [b for b in parse_blocks(result.text, SCORE_KEYS) if _score(b) is not None]
    return [
        Scorecard(
            essay_id=essay_id,
            persona=persona.name,
            panel_type=panel_type,
            criterion=block.get("CRITERION", "unknown"),
            score=_score(block),
            reasoning=block.get("REASONING", ""),
        )
        for block in blocks
    ]


def compare_pairwise(essay_a: Essay, essay_b: Essay, persona: Persona, ctx: Context) -> tuple[str | None, str]:
    """Returns (winner, reasoning); winner is "A", "B", or None when unreadable."""
    result = ctx.ask(
        "judge_pairwise", tier=persona.model_tier, persona_description=persona.description,
        essay_a_text=essay_a.text, essay_b_text=essay_b.text,
    )
    return parse_pairwise(result.text)


def compare_both_orderings(candidate: Essay, reference: Essay, persona: Persona, ctx: Context) -> dict:
    """Runs the pairwise comparison in both position orderings, since
    position bias is real - only a winner that holds in both orderings
    counts. An unreadable verdict makes the pair inconsistent.
    """
    first_winner, first_reasoning = compare_pairwise(candidate, reference, persona, ctx)
    second_winner, second_reasoning = compare_pairwise(reference, candidate, persona, ctx)

    candidate_won_first = first_winner == "A"
    candidate_won_second = second_winner == "B"
    consistent = None not in (first_winner, second_winner) and candidate_won_first == candidate_won_second
    return {
        "candidate_won_first_order": candidate_won_first,
        "candidate_won_second_order": candidate_won_second,
        "consistent": consistent,
        "candidate_wins": candidate_won_first and candidate_won_second,
        "first_reasoning": first_reasoning,
        "second_reasoning": second_reasoning,
    }
