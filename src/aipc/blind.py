"""Strips formatting before blind judging.

Structured conditions produce headings where zero-shot doesn't, which
would tell the judge (and the human rater) which condition wrote an
essay. The judge and the rating page both see normalised text only.
"""

import re

HEADING_MAX_WORDS = 12
TITLE_MAX_WORDS = 30
MARKDOWN_HEADER = re.compile(r"^\s*#+\s*")
BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _looks_like_heading(line: str) -> bool:
    """A short line with no closing punctuation reads as a title or section
    heading, even without '#' markup. A line of only '*' or '_' is a
    Markdown section break and goes the same way.
    """
    stripped = line.rstrip("*_")
    return not stripped or (len(line.split()) <= HEADING_MAX_WORDS and stripped[-1] not in ".?!:;,\"'”’)")


def _is_title(block: str) -> bool:
    """The essay's opening line, where every writing prompt puts the title:
    one line of at most 30 words that does not end with a full stop. Long
    titles and question titles are caught, which the heading rule missed
    in 21 of 98 essays by 2026-10-28; a first paragraph ends with a stop.
    """
    return "\n" not in block and len(block.split()) <= TITLE_MAX_WORDS and not block.rstrip("*_ ").endswith(".")


def normalise(text: str) -> str:
    """Plain paragraphs only: no title, headings, bullets or bold.

    Drops the title, '#' lines anywhere, and a heading-like line when it
    stands alone or opens a block.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    if blocks and _is_title(blocks[0]):
        blocks = blocks[1:]
    kept = []
    for block in blocks:
        lines = [BULLET.sub("", line).strip() for line in block.splitlines() if not MARKDOWN_HEADER.match(line)]
        lines = [line for line in lines if line]
        if lines and _looks_like_heading(lines[0]):
            lines = lines[1:]
        if lines:
            kept.append(" ".join(lines).replace("**", "").replace("__", ""))
    return "\n\n".join(kept)
