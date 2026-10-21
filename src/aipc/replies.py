"""Parsers for the plain-text reply formats the prompts ask for.

Parsers return None for an unreadable verdict rather than a default: a
judge that defaulted to "A" would look like position bias.
"""

import re

from aipc.models import FilterVerdict

NONE_MARKER = "NONE"
# An upper-case A or B may be followed by more text ("WINNER: A (first)");
# a lower-case one must end the line, so "Winner: a close call" is not a verdict.
WINNER_PATTERN = re.compile(
    r"^\W*(?i:winner)\W*(?:(?i:essay)\s+)?(?:([AB])\b|([ab])\W*$)", re.MULTILINE
)
# The first PASS or FAIL after VERDICT, unless it is the echoed "PASS or FAIL".
VERDICT_PATTERN = re.compile(r"^\W*VERDICT\W*(PASS|FAIL)\b(?!\s+or\b)", re.IGNORECASE | re.MULTILINE)


def is_none(reply: str) -> bool:
    """True when a model replied NONE: nothing new to raise. Markdown and
    a closing full stop are ignored ("**NONE**", "NONE.").
    """
    return reply.strip().strip("*_`").rstrip(".!").strip().upper() == NONE_MARKER


def parse_verdict(reply: str) -> FilterVerdict:
    """Reads the PASS/FAIL on the VERDICT line; an unreadable verdict fails,
    with the whole reply kept as the reasoning.
    """
    match = VERDICT_PATTERN.search(reply)
    _, found, reasoning = reply.partition("REASONING:")
    passed = match is not None and match.group(1).upper() == "PASS"
    return FilterVerdict(passed=passed, reasoning=reasoning.strip() if found and match else reply.strip())


def parse_blocks(reply: str, keys: set[str]) -> list[dict[str, str]]:
    """Reads blank-line-separated blocks of `KEY: value` lines, keeping only
    the given (upper-case) keys.
    """
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in reply.strip().splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, _, value = line.partition(":")
        if key.strip().upper() in keys:
            current[key.strip().upper()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def parse_winner(reply: str) -> str | None:
    """Returns "A" or "B" from the last WINNER line, or None if none names one.

    The last line, because a reasoning-first prompt puts the verdict after
    the reasoning, which may itself mention a winner.
    """
    matches = [m.group(1) or m.group(2) for m in WINNER_PATTERN.finditer(reply)]
    return matches[-1].upper() if matches else None


def parse_pairwise(reply: str) -> tuple[str | None, str]:
    """Returns (winner, reasoning) from a WINNER / REASONING reply, in
    either order.
    """
    _, _, reasoning = reply.partition("REASONING:")
    return parse_winner(reply), WINNER_PATTERN.sub("", reasoning).strip()
