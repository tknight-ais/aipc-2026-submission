"""Generation-strategy seam: turns a Decomposition into a ContentLedger
plus per-sub-question transcripts. Which strategy runs is config
(`pipeline.strategy`), not a code branch. Two implementations:

- `flat` - one direct call per sub-question, no persona debate. The
  baseline the Debate Room must beat to earn its cost.
- `debate_room` - Manager poses each sub-question, two position-defined
  debaters argue it (seeing only published turns, never each other's
  private reasoning), Manager judges convergence and synthesises the
  transcript into ledger entries.
"""

from aipc.context import Context
from aipc.models import Claim, ContentLedger, DebateTranscript, DebateTurn, Decomposition, Objection
from aipc.replies import is_none, parse_blocks

_LEDGER_BLOCK_KEYS = {"CLAIM", "SUPPORT", "OBJECTION", "REPLY", "REPLY_TYPE", "CONCESSION"}


def _blocks_to_ledger_parts(blocks: list[dict[str, str]], prefix: str) -> tuple[list[Claim], list[Objection], list[str]]:
    claims: list[Claim] = []
    objections: list[Objection] = []
    concessions: list[str] = []
    claim_idx = 0
    for block in blocks:
        if "CONCESSION" in block:
            concessions.append(block["CONCESSION"])
        if "CLAIM" not in block:
            continue
        claim_idx += 1
        claim_id = f"{prefix}_c{claim_idx}"
        claims.append(Claim(id=claim_id, statement=block["CLAIM"], support=block.get("SUPPORT", ""), role="premise"))
        if "OBJECTION" in block:
            reply_type = "decisive" if "decisive" in block.get("REPLY_TYPE", "").lower() else "concessive"
            objections.append(Objection(
                statement=block["OBJECTION"],
                target_claim_id=claim_id,
                reply=block.get("REPLY", ""),
                reply_type=reply_type,
            ))
    return claims, objections, concessions


def _assemble_ledger(thesis: str, parts: list[tuple[list[Claim], list[Objection], list[str]]]) -> ContentLedger:
    claims, objections, concessions = [], [], []
    for c, o, co in parts:
        claims.extend(c)
        objections.extend(o)
        concessions.extend(co)
    return ContentLedger(thesis=thesis, claims=claims, objections=objections, concessions=concessions, engagements=[])


def run_flat(decomposition: Decomposition, ctx: Context) -> tuple[ContentLedger, list[DebateTranscript]]:
    parts = []
    transcripts = []
    for i, sub_question in enumerate(decomposition.sub_questions):
        result = ctx.ask("ledger_flat", thesis=decomposition.thesis, sub_question=sub_question)
        parts.append(_blocks_to_ledger_parts(parse_blocks(result.text, _LEDGER_BLOCK_KEYS), prefix=f"sq{i}"))
        transcripts.append(DebateTranscript(
            sub_question=sub_question,
            turns=[DebateTurn(speaker="writer", text=result.text)],
            converged=True,
            rounds_used=1,
        ))

    return _assemble_ledger(decomposition.thesis, parts), transcripts


def _format_transcript(turns: list[DebateTurn]) -> str:
    return "\n\n".join(f"{t.speaker}: {t.text}" for t in turns) if turns else "(no turns yet)"


def _debate_one_sub_question(sub_question: str, thesis: str, min_rounds: int, max_rounds: int, ctx: Context) -> DebateTranscript:
    """Convergence is first checked after `min_rounds`: after round 1 every
    point is new by definition, yet a manager checking there tends to call
    the debate converged, leaving a single exchange.
    """
    turns: list[DebateTurn] = []
    converged = False
    rounds_used = 0

    for round_num in range(1, max_rounds + 1):
        for speaker in ("for", "against"):
            reply = ctx.ask(f"debater_{speaker}", thesis=thesis, sub_question=sub_question, transcript=_format_transcript(turns))
            turns.append(DebateTurn(speaker=speaker, text=reply.text))

        rounds_used = round_num
        if round_num < min_rounds:
            continue
        verdict = ctx.ask("debate_converge", sub_question=sub_question, transcript=_format_transcript(turns))
        if is_none(verdict.text):
            converged = True
            break

    return DebateTranscript(sub_question=sub_question, turns=turns, converged=converged, rounds_used=rounds_used)


def _synthesize(transcript: DebateTranscript, prefix: str, ctx: Context) -> tuple[list[Claim], list[Objection], list[str]]:
    result = ctx.ask("debate_synthesize", sub_question=transcript.sub_question, transcript=_format_transcript(transcript.turns))
    return _blocks_to_ledger_parts(parse_blocks(result.text, _LEDGER_BLOCK_KEYS), prefix=prefix)


def run_debate_room(decomposition: Decomposition, ctx: Context) -> tuple[ContentLedger, list[DebateTranscript]]:
    pipeline = ctx.config.pipeline
    parts = []
    transcripts = []

    for i, sub_question in enumerate(decomposition.sub_questions):
        transcript = _debate_one_sub_question(
            sub_question, decomposition.thesis, pipeline.debate_min_rounds, pipeline.debate_max_rounds, ctx,
        )
        transcripts.append(transcript)
        parts.append(_synthesize(transcript, prefix=f"sq{i}", ctx=ctx))

    return _assemble_ledger(decomposition.thesis, parts), transcripts


STRATEGIES = {
    "flat": run_flat,
    "debate_room": run_debate_room,
}


def run(decomposition: Decomposition, ctx: Context) -> tuple[ContentLedger, list[DebateTranscript]]:
    return STRATEGIES[ctx.config.pipeline.strategy](decomposition, ctx)
