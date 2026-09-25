"""In-memory text extraction engine for documents (.docx, .pdf, .html, .txt, .md).

Designed for fast, privacy-preserving local categorization. Content is read
in-memory and truncated to a representative sample (400-1000 chars) for Laya
inference, without writing intermediate text files to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Optional


@dataclass
class ExtractedDocument:
    """Represents a scanned document with extracted snippet and metadata."""

    file_path: Path
    file_name: str
    file_type: str
    text_snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_size_bytes: int = 0
    extraction_error: Optional[str] = None

    @property
    def prompt_text(self) -> str:
        """Formats filename, metadata, and extracted snippet for Laya input."""
        parts = [f"File: {self.file_name}"]
        if self.metadata.get("title"):
            parts.append(f"Title: {self.metadata['title']}")
        if self.metadata.get("author"):
            parts.append(f"Author: {self.metadata['author']}")
        if self.metadata.get("pages"):
            parts.append(f"Pages: {self.metadata['pages']}")

        parts.append("Content Snippet:")
        if self.text_snippet.strip():
            parts.append(self.text_snippet.strip())
        elif self.extraction_error:
            parts.append(f"[Could not extract text: {self.extraction_error}]")
        else:
            parts.append("[Empty document content]")

        return "\n".join(parts)


class DocumentTextExtractor:
    """Extracts in-memory text snippets from various document types."""

    SUPPORTED_EXTENSIONS = {
        ".docx": "docx",
        ".pdf": "pdf",
        ".html": "html",
        ".htm": "html",
        ".mhtml": "html",
        ".txt": "text",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rtf": "text",
    }

    def __init__(self, max_chars: int = 1200):
        self.max_chars = max_chars

    def is_supported(self, file_path: Path | str) -> bool:
        """Checks if a file extension is natively supported for text extraction."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: Path | str) -> ExtractedDocument:
        """Extracts text and metadata from a document file."""
        path = Path(file_path)

        if not path.exists():
            return ExtractedDocument(
                file_path=path,
                file_name=path.name,
                file_type="missing",
                text_snippet="",
                extraction_error="File does not exist",
            )

        file_size = path.stat().st_size
        ext = path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        try:
            if file_type == "docx":
                return self._extract_docx(path, file_size)
            elif file_type == "pdf":
                return self._extract_pdf(path, file_size)
            elif file_type == "html":
                return self._extract_html(path, file_size)
            elif file_type in ("text", "markdown"):
                return self._extract_plain_text(path, file_type, file_size)
            else:
                return ExtractedDocument(
                    file_path=path,
                    file_name=path.name,
                    file_type="binary",
                    text_snippet="",
                    file_size_bytes=file_size,
                    metadata={"extension": ext},
                )
        except Exception as exc:  # Catch extraction errors defensively
            return ExtractedDocument(
                file_path=path,
                file_name=path.name,
                file_type=file_type,
                text_snippet="",
                file_size_bytes=file_size,
                extraction_error=str(exc),
            )

    def _clean_whitespace(self, text: str) -> str:
        """Normalizes irregular whitespace and collapses excessive blank lines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def _truncate(self, text: str) -> str:
        """Truncates text up to max_chars preserving word boundaries where possible."""
        cleaned = self._clean_whitespace(text)
        if len(cleaned) <= self.max_chars:
            return cleaned
        truncated = cleaned[: self.max_chars]
        last_space = truncated.rfind(" ")
        if last_space > self.max_chars * 0.7:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _extract_docx(self, path: Path, file_size: int) -> ExtractedDocument:
        """Extracts text and metadata from Word .docx files using python-docx."""
        import docx

        doc = docx.Document(str(path))
        metadata: Dict[str, Any] = {}

        # Extract core document properties if present
        if doc.core_properties:
            if doc.core_properties.title:
                metadata["title"] = doc.core_properties.title.strip()
            if doc.core_properties.author:
                metadata["author"] = doc.core_properties.author.strip()
            if doc.core_properties.subject:
                metadata["subject"] = doc.core_properties.subject.strip()

        chunks: list[str] = []

        # Read headings and paragraphs
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                chunks.append(text)
            if sum(len(c) for c in chunks) >= self.max_chars * 1.5:
                break

        # If paragraphs were scarce, check tables (common for invoices/bills)
        if sum(len(c) for c in chunks) < 200:
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        chunks.append(" | ".join(row_texts))
                if sum(len(c) for c in chunks) >= self.max_chars * 1.5:
                    break

        combined_text = "\n".join(chunks)
        snippet = self._truncate(combined_text)

        return ExtractedDocument(
            file_path=path,
            file_name=path.name,
            file_type="docx",
            text_snippet=snippet,
            metadata=metadata,
            file_size_bytes=file_size,
        )

    def _extract_pdf(self, path: Path, file_size: int) -> ExtractedDocument:
        """Extracts text from PDF page 1 & 2 using pypdf."""
        import pypdf

        reader = pypdf.PdfReader(str(path))
        metadata: Dict[str, Any] = {"pages": len(reader.pages)}

        if reader.metadata:
            if reader.metadata.title:
                metadata["title"] = str(reader.metadata.title).strip()
            if reader.metadata.author:
                metadata["author"] = str(reader.metadata.author).strip()

        chunks: list[str] = []
        for i, page in enumerate(reader.pages[:2]):  # Scan up to first 2 pages
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text.strip())
            if sum(len(c) for c in chunks) >= self.max_chars * 1.5:
                break

        snippet = self._truncate("\n\n".join(chunks))

        return ExtractedDocument(
            file_path=path,
            file_name=path.name,
            file_type="pdf",
            text_snippet=snippet,
            metadata=metadata,
            file_size_bytes=file_size,
        )

    def _extract_html(self, path: Path, file_size: int) -> ExtractedDocument:
        """Extracts text from HTML / web snapshot files using BeautifulSoup."""
        from bs4 import BeautifulSoup

        raw_bytes = path.read_bytes()
        # Decode defensively
        text_content = ""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text_content = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        soup = BeautifulSoup(text_content, "html.parser")

        metadata: Dict[str, Any] = {}
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"].strip()

        # Remove irrelevant elements
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()

        body_text = soup.get_text(separator="\n")
        snippet = self._truncate(body_text)

        return ExtractedDocument(
            file_path=path,
            file_name=path.name,
            file_type="html",
            text_snippet=snippet,
            metadata=metadata,
            file_size_bytes=file_size,
        )

    def _extract_plain_text(self, path: Path, file_type: str, file_size: int) -> ExtractedDocument:
        """Extracts text from plain text or markdown files."""
        raw_bytes = path.read_bytes()
        text_content = ""
        for encoding in ("utf-8", "latin-1", "cp1252", "utf-16"):
            try:
                text_content = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        snippet = self._truncate(text_content)
        return ExtractedDocument(
            file_path=path,
            file_name=path.name,
            file_type=file_type,
            text_snippet=snippet,
            file_size_bytes=file_size,
        )
