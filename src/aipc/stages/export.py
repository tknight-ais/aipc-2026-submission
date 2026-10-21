"""PDF export of the submitted essay.

Markdown goes through pandoc (a system install, not a Python package) to
Typst, and the `typst` package compiles that to PDF, so no LaTeX install
is needed. The .typ source is kept next to the PDF.
"""

import subprocess
from pathlib import Path

import typst

PANDOC_ARGS = ["-s", "-t", "typst", "-V", "mainfont=Libertinus Serif", "-V", "margin.x=2.5cm", "-V", "margin.y=2.5cm"]


def to_typst(markdown_text: str) -> str:
    result = subprocess.run(["pandoc", *PANDOC_ARGS], input=markdown_text, capture_output=True, text=True, check=True)
    return result.stdout


def to_pdf(markdown_text: str, pdf_path: Path) -> Path:
    """Writes <pdf_path> and its .typ source; returns the .typ path."""
    typ_path = pdf_path.with_suffix(".typ")
    typ_path.write_text(to_typst(markdown_text))
    typst.compile(str(typ_path), output=str(pdf_path))
    return typ_path
