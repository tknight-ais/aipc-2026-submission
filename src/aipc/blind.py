"""Strips formatting before blind judging.

Structured conditions produce headings where zero-shot doesn't, which
would tell the judge (and the human rater) which condition wrote an
essay. The judge and the rating page both see normalised text only.
"""

import re

HEADING_MAX_WORDS = 12
MARKDOWN_HEADER = re.compile(r"^\s*#+\s*")
BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _looks_like_heading(line: str) -> bool:
    """A short line with no closing punctuation reads as a title or section
    heading, even without '#' markup. A line of only '*' or '_' is a
    Markdown section break and goes the same way.
    """
    stripped = line.rstrip("*_")
    return not stripped or (len(line.split()) <= HEADING_MAX_WORDS and stripped[-1] not in ".?!:;,\"'”’)")


def normalise(text: str) -> str:
    """Plain paragraphs only: no titles, headings, bullets or bold.

    Drops '#' lines anywhere, and a heading-like line when it stands alone
    or opens a block.
    """
    kept = []
    for block in re.split(r"\n\s*\n", text):
        lines = [BULLET.sub("", line).strip() for line in block.splitlines() if not MARKDOWN_HEADER.match(line)]
        lines = [line for line in lines if line]
        if lines and _looks_like_heading(lines[0]):
            lines = lines[1:]
        if lines:
            kept.append(" ".join(lines).replace("**", "").replace("__", ""))
    return "\n\n".join(kept)
