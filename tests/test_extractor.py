"""Unit tests for DocumentTextExtractor."""

from pathlib import Path
import docx
import pypdf
import pytest

from ordinale.extractor import DocumentTextExtractor, ExtractedDocument


@pytest.fixture
def extractor() -> DocumentTextExtractor:
    return DocumentTextExtractor(max_chars=300)


def test_is_supported(extractor: DocumentTextExtractor):
    assert extractor.is_supported("contract.docx")
    assert extractor.is_supported("statement.pdf")
    assert extractor.is_supported("article.html")
    assert extractor.is_supported("notes.md")
    assert extractor.is_supported("readme.txt")
    assert not extractor.is_supported("installer.exe")
    assert not extractor.is_supported("archive.zip")


def test_extract_missing_file(extractor: DocumentTextExtractor, tmp_path: Path):
    missing_path = tmp_path / "non_existent.docx"
    doc = extractor.extract(missing_path)
    assert doc.file_type == "missing"
    assert "File does not exist" in (doc.extraction_error or "")


def test_extract_plain_text_and_markdown(extractor: DocumentTextExtractor, tmp_path: Path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Meeting Notes\nDiscussed Q3 budget and CRA tax filing deadlines.", encoding="utf-8")

    doc = extractor.extract(txt_file)
    assert doc.file_type == "text"
    assert "Meeting Notes" in doc.text_snippet
    assert "CRA tax filing" in doc.text_snippet
    assert "notes.txt" in doc.prompt_text


def test_extract_html(extractor: DocumentTextExtractor, tmp_path: Path):
    html_file = tmp_path / "article.html"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Guide to University Admissions</title>
        <meta name="description" content="A comprehensive overview of college application requirements.">
    </head>
    <body>
        <script>console.log("ignore me");</script>
        <h1>Undergraduate Admissions 2026</h1>
        <p>Submit your high school transcripts and recommendation letters by November 15.</p>
    </body>
    </html>
    """
    html_file.write_text(html_content, encoding="utf-8")

    doc = extractor.extract(html_file)
    assert doc.file_type == "html"
    assert doc.metadata.get("title") == "Guide to University Admissions"
    assert "Undergraduate Admissions 2026" in doc.text_snippet
    assert "ignore me" not in doc.text_snippet
    assert "Guide to University Admissions" in doc.prompt_text


def test_extract_docx(extractor: DocumentTextExtractor, tmp_path: Path):
    docx_file = tmp_path / "sample_assignment.docx"
    doc_builder = docx.Document()
    doc_builder.core_properties.title = "CS101 Lab 3 Assignment"
    doc_builder.core_properties.author = "Jane Doe"
    doc_builder.add_heading("Lab 3: Binary Search Trees", level=1)
    doc_builder.add_paragraph("Implement insert, delete, and in-order traversal in Python.")

    # Add table with some details
    table = doc_builder.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Deliverable"
    table.cell(0, 1).text = "Weight"
    table.cell(1, 0).text = "bst.py"
    table.cell(1, 1).text = "10%"

    doc_builder.save(str(docx_file))

    doc = extractor.extract(docx_file)
    assert doc.file_type == "docx"
    assert doc.metadata.get("title") == "CS101 Lab 3 Assignment"
    assert doc.metadata.get("author") == "Jane Doe"
    assert "Binary Search Trees" in doc.text_snippet
    assert "insert, delete" in doc.text_snippet
    assert "bst.py" in doc.text_snippet
    assert "Jane Doe" in doc.prompt_text


def test_extract_pdf(extractor: DocumentTextExtractor, tmp_path: Path):
    pdf_file = tmp_path / "cra_notice.pdf"

    # Generate minimal valid PDF with pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # pypdf Writer can set metadata
    writer.add_metadata({
        "/Title": "Notice of Assessment 2025",
        "/Author": "Canada Revenue Agency",
    })
    with open(pdf_file, "wb") as f:
        writer.write(f)

    doc = extractor.extract(pdf_file)
    assert doc.file_type == "pdf"
    assert doc.metadata.get("pages") == 1
    assert "Notice of Assessment 2025" in (doc.metadata.get("title") or "")


def test_extract_binary_fallback(extractor: DocumentTextExtractor, tmp_path: Path):
    bin_file = tmp_path / "installer.exe"
    bin_file.write_bytes(b"\x4D\x5A\x90\x00\x03\x00")

    doc = extractor.extract(bin_file)
    assert doc.file_type == "binary"
    assert doc.file_size_bytes == 6
    assert doc.metadata.get("extension") == ".exe"


def test_extract_rtf(extractor: DocumentTextExtractor, tmp_path: Path):
    rtf_file = tmp_path / "agreement.rtf"
    rtf_content = r"""{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Arial;}}
{\*\generator Riched20 10.0.19041}\viewkind4\uc1
\pard\sa200\sl276\slmult1\b\f0\fs24 Residential Tenancy Agreement\b0\par
This agreement is entered into on January 1, 2026.\par
Monthly rent: \$1,800.\par
}"""
    rtf_file.write_text(rtf_content, encoding="utf-8")

    doc = extractor.extract(rtf_file)
    assert doc.file_type == "rtf"
    assert "Residential Tenancy Agreement" in doc.text_snippet
    assert "Monthly rent: $1,800." in doc.text_snippet
    assert r"\rtf1" not in doc.text_snippet
    assert r"\fonttbl" not in doc.text_snippet
    assert r"\generator" not in doc.text_snippet
    assert "agreement.rtf" in doc.prompt_text


def test_extract_pdf_corrupt_xref_no_stderr(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Corrupt startxref pointer is handled cleanly with zero stderr pollution."""
    pdf_file = tmp_path / "corrupt_xref.pdf"
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000060 00000 n \n"
        b"0000000117 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n"
        b"99999\n"
        b"%%EOF\n"
    )
    pdf_file.write_bytes(content)

    doc = extractor.extract(pdf_file)
    captured = capsys.readouterr()

    assert captured.err == ""
    assert doc.file_type == "pdf"
    assert doc.extraction_error is None
    assert "extraction_warnings" in doc.metadata
    assert any("startxref" in w.lower() for w in doc.metadata["extraction_warnings"])


def test_extract_pdf_truncated_no_stderr(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Truncated corrupt PDF sets extraction_error without stderr pollution."""
    pdf_file = tmp_path / "truncated.pdf"
    pdf_file.write_bytes(b"%PDF-1.5\n%incomplete file without valid objects\n")

    doc = extractor.extract(pdf_file)
    captured = capsys.readouterr()

    assert captured.err == ""
    assert doc.file_type == "pdf"
    assert doc.extraction_error is not None
    assert "corrupt" in doc.extraction_error.lower() or "unreadable" in doc.extraction_error.lower()


def test_extract_pdf_empty_file_no_stderr(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Empty 0-byte PDF is cleanly handled without stderr output."""
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"")

    doc = extractor.extract(pdf_file)
    captured = capsys.readouterr()

    assert captured.err == ""
    assert doc.file_type == "pdf"
    assert doc.extraction_error == "Empty or zero-byte PDF file"


def test_extract_pdf_encrypted_no_stderr(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Password-encrypted PDF without empty password access reports clean error without crashing or stderr output."""
    pdf_file = tmp_path / "encrypted.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt(user_password="secret_password_123")
    with open(pdf_file, "wb") as f:
        writer.write(f)

    doc = extractor.extract(pdf_file)
    captured = capsys.readouterr()

    assert captured.err == ""
    assert doc.file_type == "pdf"
    assert doc.extraction_error == "Encrypted or password-protected PDF"


def test_extract_pdf_concurrent_corrupt_no_stderr(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Concurrent extraction of corrupt PDFs across threads produces zero stderr leaks."""
    from concurrent.futures import ThreadPoolExecutor

    files = []
    for i in range(10):
        f = tmp_path / f"corrupt_{i}.pdf"
        f.write_bytes(f"%PDF-1.4\ncorrupt junk {i}\n%%EOF".encode("utf-8"))
        files.append(f)

    with ThreadPoolExecutor(max_workers=4) as executor:
        docs = list(executor.map(extractor.extract, files))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(docs) == 10
    for doc in docs:
        assert doc.extraction_error is not None


def test_extract_pdf_partial_page_corruption(
    extractor: DocumentTextExtractor, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """If page 2 fails with a corrupt stream error, page 1 text is preserved cleanly."""
    import pypdf
    from unittest.mock import MagicMock

    pdf_file = tmp_path / "two_pages.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    # Mock PdfReader to return page 1 with text and page 2 that raises PdfStreamError
    def mock_pdf_reader(stream, strict=False):
        reader = MagicMock()
        reader.is_encrypted = False
        reader.metadata = None
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 Successful Invoicing Text"
        page2 = MagicMock()
        page2.extract_text.side_effect = pypdf.errors.PdfStreamError("Corrupt stream object")
        reader.pages = [page1, page2]
        return reader

    monkeypatch.setattr(pypdf, "PdfReader", mock_pdf_reader)

    doc = extractor.extract(pdf_file)
    captured = capsys.readouterr()

    assert captured.err == ""
    assert doc.file_type == "pdf"
    assert "Page 1 Successful Invoicing Text" in doc.text_snippet
    assert doc.extraction_error is None
    assert "extraction_warnings" in doc.metadata
    assert any("Corrupt stream object" in w for w in doc.metadata["extraction_warnings"])
