"""Parsers for the plain-text reply formats the prompts ask for.

Parsers return None for an unreadable verdict rather than a default: a
judge that defaulted to "A" would look like position bias.
"""

import re

from aipc.models import FilterVerdict

NONE_MARKER = "NONE"
# An upper-case A or B may be followed by more text ("WINNER: A (first)");
# a lower-case one must end the line, so "Winner: a close call" is not a verdict.
WINNER_PATTERN = re.compile(r"^\W*(?i:winner)\W*(?:(?i:essay)\s+)?(?:([AB])\b|([ab])\W*$)", re.MULTILINE)
# The first PASS or FAIL after VERDICT, unless it is the echoed "PASS or FAIL".
VERDICT_PATTERN = re.compile(r"^\W*VERDICT\W*(PASS|FAIL)\b(?!\s+or\b)", re.IGNORECASE | re.MULTILINE)
# A `KEY: value` line, allowing the decoration models add around the key:
# markdown emphasis, a bullet or number before it, a number after it
# ("**CLAIM 1:**", "- REPLY TYPE:").
KEY_LINE = re.compile(r"^[\s>*_#-]*(?:\d+[.)]\s*)?[*_]*([A-Za-z][A-Za-z_ ]*?)\s*\d*\s*[*_]*:[*_]*\s*(.*)$")
# A bullet or a number opening a list item: "- ", "* ", "1. ", "**2.** ".
LIST_MARKER = re.compile(r"^\s*(?:[-*+\u2022]\s+|\**\d+[.)]\**\s+)")


def is_none(reply: str) -> bool:
    """True when a model replied NONE: nothing new to raise. Only the first
    line is read, so an explanation after it doesn't turn NONE into an
    objection. Markdown and a closing full stop are ignored ("**NONE**",
    "NONE.").
    """
    first_line = reply.strip().partition("\n")[0]
    return first_line.strip().strip("*_`").rstrip(".!").strip().upper() == NONE_MARKER


def parse_verdict(reply: str) -> FilterVerdict:
    """Reads the PASS/FAIL on the VERDICT line; an unreadable verdict fails,
    with the whole reply kept as the reasoning.
    """
    match = VERDICT_PATTERN.search(reply)
    _, found, reasoning = reply.partition("REASONING:")
    passed = match is not None and match.group(1).upper() == "PASS"
    return FilterVerdict(passed=passed, reasoning=reasoning.strip() if found and match else reply.strip())


def _key_value(line: str, keys: set[str]) -> tuple[str | None, str]:
    """(key, value) for a line naming one of `keys`, else (None, line)."""
    match = KEY_LINE.match(line)
    key = match.group(1).strip().upper().replace(" ", "_") if match else None
    if match is None or key not in keys:
        return None, line.strip()
    return key, match.group(2).strip()


def parse_entries(reply: str, keys: set[str], start: str) -> list[dict[str, str]]:
    """Reads `KEY: value` lines into entries, keeping only the given
    (upper-case) keys. A `start` key opens a new entry, as does a key the
    current entry already has. Blank lines do not: models put them inside
    entries. A line with no key continues the value above it, up to the
    next blank line.
    """
    entries: list[dict[str, str]] = []
    last_key = None
    for line in reply.splitlines():
        if not line.strip():
            last_key = None
            continue
        key, value = _key_value(line, keys)
        if key is None:
            if last_key is not None:
                entries[-1][last_key] = f"{entries[-1][last_key]} {value}".strip()
            continue
        if key == start or not entries or key in entries[-1]:
            entries.append({})
        entries[-1][key] = value
        last_key = key
    return entries


def list_items(reply: str) -> list[str]:
    """The items of a one-per-line reply. Bullets, numbering, bold and
    quotation marks around an item are dropped, and so is a line ending in
    a colon, which introduces the list rather than belonging to it.
    """
    items = [LIST_MARKER.sub("", line).strip().strip('*"\u201c\u201d').strip() for line in reply.splitlines()]
    return [item for item in items if item and not item.endswith(":")]


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
