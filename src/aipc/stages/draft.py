"""Drafter / Critics / Reviser stage: turns a ContentLedger into a
finished Essay, then runs a parallel-critic revision loop against it.

The Drafter owns structure - the ledger supplies claims, objections and
concessions only, never section order or headings. Critics run in
parallel per round rather than as sequential rounds, since revision gains
concentrate in the first pass. The loop stops when
a round's critics raise nothing new, or after a hard cap of rounds.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from aipc.context import Context
from aipc.models import ContentLedger, Critique, Essay, FidelityReview, RevisionRound, split_bibliography
from aipc.replies import is_none

# A deliberately simple lexical heuristic for boldness erosion, not a
# claim of semantic precision - cheap to log per round, which is the
# point.
HEDGE_WORDS = [
    "perhaps",
    "arguably",
    "possibly",
    "somewhat",
    "to some extent",
    "it could be argued",
    "it might be",
    "it may be",
    "one might think",
    "it seems",
    "seemingly",
    "presumably",
    "plausibly",
    "may well",
    "might well",
    "could be seen as",
    "in some sense",
]
# Whole words only: "possibly" must not match inside "impossibly".
HEDGE_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(phrase) for phrase in HEDGE_WORDS) + r")\b")


def hedge_density(text: str) -> float:
    """Hedge-phrase occurrences per 1,000 words of the body (the reference
    list is excluded from both counts).
    """
    body = split_bibliography(text)[0]
    words = len(body.split())
    if words == 0:
        return 0.0
    return len(HEDGE_PATTERN.findall(body.lower())) / words * 1000


def essay_requirements(ctx: Context) -> str:
    """The instructions every condition's writing prompts share - length,
    format, title, references, citation policy - so conditions differ only
    in the component under test. Separate rules per prompt would confound
    the ladder.
    """
    word_min, word_max = ctx.config.limits.target_words
    return ctx.render("essay_requirements", word_min=word_min, word_max=word_max)


def draft_essay(ledger: ContentLedger, ctx: Context) -> Essay:
    """The Drafter's first pass: ledger in, essay out, no revision."""
    result = ctx.ask("draft", ledger_json=ledger.model_dump_json(indent=2), requirements=essay_requirements(ctx))
    return Essay.from_text(result.text, thesis=ledger.thesis)


def run_critics(essay: Essay, thesis: str, previously_raised: list[str], ctx: Context) -> list[Critique]:
    """Runs all critics for one round in parallel, each blind to the
    others' output this round - only to objections already published
    from earlier rounds. Each critic gets its own seed.

    The prompt is rendered once, on the calling thread, because every
    critic sees the same text; each critic's call is recorded from its own
    thread, through the store's write lock.
    """
    prompt = ctx.render(
        "critic",
        thesis=thesis,
        essay_text=essay.text,
        previously_raised="\n\n".join(previously_raised) or "(none yet)",
    )

    def critique(critic_id: int) -> Critique:
        seed = ctx.config.defaults.seed + critic_id
        result = ctx.llm.call(tier=ctx.config.defaults.tier, prompt=prompt, seed=seed, prompt_name="critic")
        return Critique(critic_id=critic_id, text=result.text.strip())

    with ThreadPoolExecutor(max_workers=ctx.config.pipeline.critics) as pool:
        return list(pool.map(critique, range(ctx.config.pipeline.critics)))


def critique_round(
    essay: Essay, thesis: str, previously_raised: list[str], round_num: int, ctx: Context
) -> tuple[RevisionRound, list[str]]:
    """One round of critics. Returns the round's record and its new
    objections (empty when every critic replied NONE).
    """
    critiques = run_critics(essay, thesis, previously_raised, ctx)
    objections = [c.text for c in critiques if not is_none(c.text)]
    record = RevisionRound(
        round_num=round_num,
        critiques=critiques,
        has_new_objection=bool(objections),
        word_count=essay.word_count,
        hedge_density=hedge_density(essay.text),
    )
    return record, objections


def revise(essay: Essay, ledger: ContentLedger, objections: list[str], ctx: Context) -> Essay:
    result = ctx.ask(
        "revise",
        ledger_json=ledger.model_dump_json(indent=2),
        essay_text=essay.text,
        objections_text="\n\n".join(objections),
        requirements=essay_requirements(ctx),
    )
    return Essay.from_text(result.text, thesis=ledger.thesis)


def revise_free(essay: Essay, objections: list[str], ctx: Context) -> Essay:
    """revise() without a ledger, for essays that were never built from one."""
    result = ctx.ask(
        "revise_free",
        essay_text=essay.text,
        objections_text="\n\n".join(objections),
        requirements=essay_requirements(ctx),
    )
    return Essay.from_text(result.text, thesis=essay.thesis)


def check_fidelity(essay: Essay, ledger: ContentLedger, ctx: Context) -> FidelityReview:
    result = ctx.ask("fidelity", ledger_json=ledger.model_dump_json(indent=2), essay_text=essay.text)
    dropped_part, _, claims_part = result.text.partition("UNSUPPORTED CLAIMS:")
    dropped = dropped_part.replace("DROPPED OBJECTIONS:", "").strip()
    return FidelityReview(dropped_objections=dropped, unsupported_claims=claims_part.strip())


def run(ledger: ContentLedger, ctx: Context) -> tuple[Essay, list[RevisionRound], FidelityReview]:
    """Full Drafter -> Critics -> Reviser loop, then a ledger-fidelity
    pass over the final essay. Stops once every critic in a round
    returns NONE; otherwise runs to the configured hard cap.
    """
    essay = draft_essay(ledger, ctx)
    rounds: list[RevisionRound] = []
    raised_so_far: list[str] = []

    for round_num in range(1, ctx.config.pipeline.max_revision_rounds + 1):
        record, objections = critique_round(essay, ledger.thesis, raised_so_far, round_num, ctx)
        rounds.append(record)
        if not objections:
            break
        raised_so_far.extend(objections)
        essay = revise(essay, ledger, objections, ctx)
        essay.rounds_used = round_num

    return essay, rounds, check_fidelity(essay, ledger, ctx)
