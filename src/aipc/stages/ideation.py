"""Ideation stage: generates candidate theses in a cold batch (no
literature context) and an informed batch (with it), then screens the
pooled candidates - significance first (cheap, no retrieval), novelty
second (spent only on candidates already judged worth arguing).

Each candidate is its own call with no shared context, so no candidate
sees another. That does not stop convergence: every call sends the same
prompt, so the candidates are samples from one distribution and tend to
converge. Claude models ignore the seed, so it does not vary them either.
"""

import re
from typing import Literal

from aipc.context import Context
from aipc.models import FilterVerdict, LiteraturePack, ScreenedThesis, ThesisCandidate
from aipc.replies import parse_verdict

THESIS_KEY = re.compile(r"[*_#\s]*THESIS[*_]*:[*_]*")
RATIONALE_KEY = re.compile(r"[*_#\s]*RATIONALE[*_]*:[*_]*")


def _parse_thesis(text: str) -> tuple[str, str]:
    """(thesis, rationale): the text after THESIS and after RATIONALE, with
    any markdown around either key dropped.
    """
    thesis_part, _, rationale_part = RATIONALE_KEY.sub("RATIONALE:", text, count=1).partition("RATIONALE:")
    return THESIS_KEY.sub("", thesis_part, count=1).strip(), rationale_part.strip()


def _sources_text(lit_pack: LiteraturePack) -> str:
    return "\n".join(lit_pack.sources) or "(no results)"


def _generate(
    prompt_name: str, origin: Literal["cold", "informed"], n: int, ctx: Context, **fields
) -> list[ThesisCandidate]:
    candidates = []
    for _ in range(n):
        thesis, rationale = _parse_thesis(ctx.ask(prompt_name, **fields).text)
        candidates.append(ThesisCandidate(claim=thesis, origin=origin, rationale=rationale))
    return candidates


def generate_cold(topic: str, n: int, ctx: Context) -> list[ThesisCandidate]:
    return _generate("ideation_cold", "cold", n, ctx, topic=topic)


def generate_informed(topic: str, lit_pack: LiteraturePack, n: int, ctx: Context) -> list[ThesisCandidate]:
    return _generate(
        "ideation_informed",
        "informed",
        n,
        ctx,
        topic=topic,
        background=lit_pack.background,
        sources=_sources_text(lit_pack),
    )


def check_significance(candidate: ThesisCandidate, ctx: Context) -> FilterVerdict:
    return parse_verdict(ctx.ask("significance_filter", thesis=candidate.claim, rationale=candidate.rationale).text)


def check_novelty(candidate: ThesisCandidate, lit_pack: LiteraturePack, ctx: Context) -> FilterVerdict:
    return parse_verdict(ctx.ask("novelty_filter", thesis=candidate.claim, sources=_sources_text(lit_pack)).text)


def screen(candidates: list[ThesisCandidate], lit_pack: LiteraturePack, ctx: Context) -> list[ScreenedThesis]:
    """A dry run's placeholder verdict never passes, so a dry run checks
    novelty for every candidate: its cost estimate stays an upper bound.
    """
    screened = []
    for candidate in candidates:
        significance = check_significance(candidate, ctx)
        novelty = check_novelty(candidate, lit_pack, ctx) if significance.passed or ctx.config.dry_run else None
        screened.append(ScreenedThesis(candidate=candidate, significance=significance, novelty=novelty))
    return screened


def run(topic: str, lit_pack: LiteraturePack, ctx: Context) -> list[ScreenedThesis]:
    n = ctx.config.pipeline.ideation_candidates
    cold = generate_cold(topic, n, ctx)
    informed = generate_informed(topic, lit_pack, n, ctx)
    return screen(cold + informed, lit_pack, ctx)
