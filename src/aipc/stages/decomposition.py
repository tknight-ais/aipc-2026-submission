"""Decomposition stage: splits a thesis into sub-questions that structure
the essay's whole argument - its own step because it happens once and
caps the essay's shape. A validity gate screens for
*structural* validity only (coverage, overlap, answerability), then a
selection call picks among the structurally valid survivors.

Every call uses the run's generator tier (`defaults.tier`), not a
stronger model: the component ladder needs one generator across all
conditions, or C and D would differ from A and B in model as well as
method.
"""

import re

from aipc.context import Context
from aipc.models import Decomposition, DecompositionSelection, FilterVerdict, ScreenedDecomposition
from aipc.replies import list_items, parse_verdict

CHOICE_PATTERN = re.compile(r"CHOICE\W*(\d+)", re.IGNORECASE)
# decompose.md asks for 3-5 sub-questions; a dry run plans for the most,
# so its cost estimate for C and D stays an upper bound.
MAX_SUB_QUESTIONS = 5


def _sub_questions(reply: str, dry_run: bool) -> list[str]:
    if dry_run:
        return [f"[dry-run sub-question {i + 1}]" for i in range(MAX_SUB_QUESTIONS)]
    return list_items(reply)


def generate_candidates(thesis: str, n: int, ctx: Context) -> list[Decomposition]:
    return [
        Decomposition(
            thesis=thesis, sub_questions=_sub_questions(ctx.ask("decompose", thesis=thesis).text, ctx.config.dry_run)
        )
        for _ in range(n)
    ]


def check_validity(decomposition: Decomposition, ctx: Context) -> FilterVerdict:
    result = ctx.ask("validity_gate", thesis=decomposition.thesis, sub_questions="\n".join(decomposition.sub_questions))
    return parse_verdict(result.text)


def screen(candidates: list[Decomposition], ctx: Context) -> list[ScreenedDecomposition]:
    return [
        ScreenedDecomposition(decomposition=candidate, validity=check_validity(candidate, ctx))
        for candidate in candidates
    ]


def select(thesis: str, survivors: list[Decomposition], ctx: Context) -> DecompositionSelection:
    """Picks among structurally valid survivors only. Callers must not
    pass a candidate that failed the validity gate.
    """
    candidates_text = "\n\n".join(f"{i + 1}. " + " / ".join(d.sub_questions) for i, d in enumerate(survivors))
    result = ctx.ask("decomposition_select", thesis=thesis, candidates=candidates_text)

    if ctx.config.dry_run:
        # The dry-run placeholder text can't satisfy "CHOICE: <n>" - the
        # call still happened, for an honest cost estimate, but there is
        # nothing real to parse.
        return DecompositionSelection(chosen_index=0, reasoning=result.text)

    match = CHOICE_PATTERN.search(result.text)
    if match is None or not 1 <= int(match.group(1)) <= len(survivors):
        raise ValueError(f"unreadable or out-of-range decomposition choice: {result.text[:200]!r}")
    _, _, reasoning_part = result.text.partition("REASONING:")
    return DecompositionSelection(chosen_index=int(match.group(1)) - 1, reasoning=reasoning_part.strip())


def run(thesis: str, n: int, ctx: Context) -> tuple[list[ScreenedDecomposition], Decomposition]:
    """Returns every screened candidate (for inspection) plus the
    selected decomposition. Raises if none pass the validity gate -
    that is a real failure, not a case to silently fall back from.
    """
    candidates = generate_candidates(thesis, n, ctx)
    screened = screen(candidates, ctx)

    survivors = [s.decomposition for s in screened if s.validity.passed]
    if not survivors:
        if ctx.config.dry_run:
            # The dry-run placeholder text can't satisfy "VERDICT: PASS"
            # either, so nothing survives the gate - fall back to the
            # raw candidates so select() still runs for a cost estimate.
            survivors = candidates
        else:
            raise RuntimeError(f"no decomposition passed the validity gate for thesis: {thesis!r}")

    selection = select(thesis, survivors, ctx)
    return screened, survivors[selection.chosen_index]
