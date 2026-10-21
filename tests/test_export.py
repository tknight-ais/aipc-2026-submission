import shutil

import pytest

from aipc.stages import export

needs_pandoc = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")


@needs_pandoc
def test_to_pdf_writes_pdf_and_typst_source(tmp_path):
    pdf = tmp_path / "essay.pdf"
    typ = export.to_pdf("# A Title\n\nFirst paragraph.\n\nSecond, with *emphasis*.", pdf)

    assert pdf.read_bytes().startswith(b"%PDF")
    assert "First paragraph." in typ.read_text()
