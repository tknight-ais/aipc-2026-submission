"""Neutral pairwise judge for the component ladder: one model, no persona.

The persona panel in judge.py multiplies cost and was never validated,
so the ladder uses a single neutral judge instead. Its prompt is
`judge_prompt` in config.
"""

from aipc.context import Context
from aipc.replies import parse_pairwise


def compare(text_a: str, text_b: str, judge_tier: str, seed: int, ctx: Context) -> tuple[str | None, str]:
    """Returns (winner, reasoning) for one ordering of a pair; winner is
    "A", "B", or None when the reply names neither.
    """
    result = ctx.ask(ctx.config.judge_prompt, tier=judge_tier, seed=seed, essay_a_text=text_a, essay_b_text=text_b)
    return parse_pairwise(result.text)
