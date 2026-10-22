"""PDF export of the submitted essay.

Markdown goes through pandoc (a system install, not a Python package) to
Typst, and the `typst` package compiles that to PDF, so no LaTeX install
is needed. The .typ source is kept next to the PDF.

Essays are plain text by instruction: a title line, paragraphs, then
"References" with one entry per line. Pandoc would read that title as a
paragraph and run the entries together into one, so the structure is
made explicit first. The words themselves are not changed.
"""

import subprocess
from pathlib import Path

import typst

from aipc.models import split_bibliography

PANDOC_ARGS = ["-s", "-t", "typst", "-V", "mainfont=Libertinus Serif", "-V", "margin.x=2.5cm", "-V", "margin.y=2.5cm"]
# A first line this short is the title the essay requirements ask for;
# a longer one is the essay's opening paragraph.
TITLE_MAX_WORDS = 25


def to_markdown(text: str) -> tuple[str | None, str]:
    """(title, markdown): the first line as the title if it is short enough
    to be one, the body as it is, and the references under a heading, one
    paragraph per entry.
    """
    first_line, _, rest = text.strip().partition("\n")
    title = first_line.strip("#* ") if len(first_line.split()) <= TITLE_MAX_WORDS else None
    body, bibliography = split_bibliography(rest if title else text)
    entries = [line.strip() for line in bibliography.splitlines() if line.strip()]
    references = "\n\n# References\n\n" + "\n\n".join(entries) if entries else ""
    return title, body.strip() + references + "\n"


def to_typst(text: str) -> str:
    title, markdown = to_markdown(text)
    title_args = ["-M", f"title={title}"] if title else []
    result = subprocess.run(
        ["pandoc", *PANDOC_ARGS, *title_args], input=markdown, capture_output=True, text=True, check=True
    )
    return result.stdout


def to_pdf(text: str, pdf_path: Path) -> Path:
    """Writes <pdf_path> and its .typ source; returns the .typ path."""
    typ_path = pdf_path.with_suffix(".typ")
    typ_path.write_text(to_typst(text))
    typst.compile(str(typ_path), output=str(pdf_path))
    return typ_path
