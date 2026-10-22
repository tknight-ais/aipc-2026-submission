import shutil

import pytest

from aipc.stages import export

needs_pandoc = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")

ESSAY = (
    "A Title: On Things\n\n"
    "First paragraph, with *emphasis*.\n\n"
    "Second paragraph.\n\n"
    "References\n\n"
    "Parfit, Derek. 1984. Reasons and Persons. Oxford University Press.\n"
    "Frankfurt, Harry. 1971. Freedom of the Will and the Concept of a Person. Journal of Philosophy 68(1).\n"
)


def test_to_markdown_makes_the_title_heading_and_entries_explicit():
    title, markdown = export.to_markdown(ESSAY)

    assert title == "A Title: On Things"
    assert markdown.startswith("First paragraph")
    assert "\n\n# References\n\nParfit, Derek. 1984." in markdown
    assert "Oxford University Press.\n\nFrankfurt, Harry." in markdown


def test_to_markdown_leaves_an_untitled_essay_whole():
    opening = " ".join(["word"] * 40) + "."
    title, markdown = export.to_markdown(f"{opening}\n\nMore.")
    assert title is None
    assert markdown == f"{opening}\n\nMore.\n"


@needs_pandoc
def test_to_pdf_writes_pdf_and_typst_source(tmp_path):
    pdf = tmp_path / "essay.pdf"
    typ = export.to_pdf(ESSAY, pdf).read_text()

    assert pdf.read_bytes().startswith(b"%PDF")
    assert "title: [A Title: On Things]" in typ
    assert "Oxford University Press.\n\nFrankfurt, Harry." in typ
