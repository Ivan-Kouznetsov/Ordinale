"""In-memory text extraction engine for documents (.docx, .pdf, .html, .txt, .md).

Designed for fast, privacy-preserving local categorization. Content is read
in-memory and truncated to a representative sample (400-1000 chars) for Laya
inference, without writing intermediate text files to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterator, Optional
import warnings

_pdf_diagnostic_context: ContextVar[Optional[list[str]]] = ContextVar(
    "_pdf_diagnostic_context", default=None
)


class _PypdfLogCaptureHandler(logging.Handler):
    """Captures log records emitted by pypdf into the active context/thread."""

    def emit(self, record: logging.LogRecord) -> None:
        ctx = _pdf_diagnostic_context.get()
        if ctx is not None:
            ctx.append(record.getMessage())


def _ensure_pypdf_logging_configured() -> None:
    pypdf_logger = logging.getLogger("pypdf")
    pypdf_logger.setLevel(logging.WARNING)
    pypdf_logger.propagate = False
    if not any(isinstance(h, _PypdfLogCaptureHandler) for h in pypdf_logger.handlers):
        pypdf_logger.addHandler(_PypdfLogCaptureHandler())


_ensure_pypdf_logging_configured()


@contextmanager
def capture_pdf_diagnostics() -> Iterator[list[str]]:
    """Context manager that suppresses stderr/warnings and captures diagnostics during PDF parsing."""
    _ensure_pypdf_logging_configured()
    diagnostics: list[str] = []
    token = _pdf_diagnostic_context.set(diagnostics)
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            yield diagnostics
            for w in caught_warnings:
                msg = str(w.message)
                if msg and msg not in diagnostics:
                    diagnostics.append(msg)
    finally:
        _pdf_diagnostic_context.reset(token)



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
    front_page_text: str = ""
    clean_filename: str = ""
    nlp_entities: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Formats filename, metadata, and extracted snippet for Laya input."""
        parts = [f"File: {self.file_name}"]
        if self.clean_filename and self.clean_filename != Path(self.file_name).stem:
            parts.append(f"Clean Title: {self.clean_filename}")
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
        ".rtf": "rtf",
    }

    def __init__(
        self,
        max_chars: int = 1200,
        supported_extensions: Optional[Dict[str, str] | Any] = None,
    ):
        self.max_chars = max_chars
        if supported_extensions is None:
            self.supported_extensions = dict(self.SUPPORTED_EXTENSIONS)
        elif isinstance(supported_extensions, dict):
            self.supported_extensions = {
                (k if k.startswith(".") else f".{k}").lower().strip(): v
                for k, v in supported_extensions.items()
            }
        else:
            self.supported_extensions = {}
            for ext in supported_extensions:
                clean = ext.strip().lower()
                if not clean:
                    continue
                if not clean.startswith("."):
                    clean = f".{clean}"
                self.supported_extensions[clean] = self.SUPPORTED_EXTENSIONS.get(clean, "text")

    def is_supported(self, file_path: Path | str) -> bool:
        """Checks if a file extension is natively supported for text extraction."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

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
        file_type = self.supported_extensions.get(ext, "unknown")

        try:
            if file_type == "docx":
                doc = self._extract_docx(path, file_size)
            elif file_type == "pdf":
                doc = self._extract_pdf(path, file_size)
            elif file_type == "html":
                doc = self._extract_html(path, file_size)
            elif file_type == "rtf":
                doc = self._extract_rtf(path, file_size)
            elif file_type in ("text", "markdown"):
                doc = self._extract_plain_text(path, file_type, file_size)
            else:
                doc = ExtractedDocument(
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

        from ordinale.nlp import DocumentStructuralSegmenter, FilenamePreprocessor

        doc.clean_filename = FilenamePreprocessor.clean_filename(path.name)
        if not doc.front_page_text and doc.text_snippet:
            doc.front_page_text = DocumentStructuralSegmenter.segment_text(doc.text_snippet)["front_page"]

        return doc

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
        """Extracts text from PDF page 1 & 2 using pypdf defensively without stderr pollution."""
        import pypdf
        from pypdf.errors import (
            EmptyFileError,
            FileNotDecryptedError,
            PdfReadError,
            PdfStreamError,
            WrongPasswordError,
        )

        with capture_pdf_diagnostics() as diagnostics:
            # 1. Open reader defensively
            try:
                reader = pypdf.PdfReader(str(path), strict=False)
            except EmptyFileError:
                return ExtractedDocument(
                    file_path=path,
                    file_name=path.name,
                    file_type="pdf",
                    text_snippet="",
                    file_size_bytes=file_size,
                    extraction_error="Empty or zero-byte PDF file",
                )
            except (FileNotDecryptedError, WrongPasswordError):
                return ExtractedDocument(
                    file_path=path,
                    file_name=path.name,
                    file_type="pdf",
                    text_snippet="",
                    file_size_bytes=file_size,
                    extraction_error="Encrypted or password-protected PDF",
                )
            except Exception as exc:
                err_detail = f": {exc}" if str(exc) else ""
                diag_msg = f" ({'; '.join(diagnostics)})" if diagnostics else ""
                return ExtractedDocument(
                    file_path=path,
                    file_name=path.name,
                    file_type="pdf",
                    text_snippet="",
                    file_size_bytes=file_size,
                    extraction_error=f"Corrupt or unreadable PDF structure{err_detail}{diag_msg}",
                )

            # Check if encrypted and cannot be decrypted
            if getattr(reader, "is_encrypted", False):
                try:
                    decrypted = reader.decrypt("")
                    if decrypted == 0:
                        return ExtractedDocument(
                            file_path=path,
                            file_name=path.name,
                            file_type="pdf",
                            text_snippet="",
                            file_size_bytes=file_size,
                            extraction_error="Encrypted or password-protected PDF",
                        )
                except Exception:
                    return ExtractedDocument(
                        file_path=path,
                        file_name=path.name,
                        file_type="pdf",
                        text_snippet="",
                        file_size_bytes=file_size,
                        extraction_error="Encrypted or password-protected PDF",
                    )

            # 2. Extract metadata defensively
            metadata: Dict[str, Any] = {}
            try:
                metadata["pages"] = len(reader.pages)
            except Exception:
                metadata["pages"] = 0

            try:
                if reader.metadata:
                    if reader.metadata.title:
                        metadata["title"] = str(reader.metadata.title).strip()
                    if reader.metadata.author:
                        metadata["author"] = str(reader.metadata.author).strip()
            except Exception:
                pass

            # 3. Read pages defensively (up to first 2 pages)
            chunks: list[str] = []
            page_errors: list[str] = []
            pages_to_scan = []
            try:
                pages_to_scan = reader.pages[:2]
            except Exception as exc:
                page_errors.append(f"Could not load page list: {exc}")

            for page in pages_to_scan:
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        chunks.append(page_text.strip())
                    if sum(len(c) for c in chunks) >= self.max_chars * 1.5:
                        break
                except Exception as exc:
                    page_errors.append(str(exc))

            combined_text = "\n\n".join(chunks)
            snippet = self._truncate(combined_text)

            all_warnings = list(diagnostics)
            if page_errors:
                all_warnings.extend(f"Page error: {e}" for e in page_errors)
            if all_warnings:
                metadata["extraction_warnings"] = all_warnings

            extraction_error = None
            if not snippet:
                if page_errors:
                    extraction_error = f"Corrupt or unreadable PDF content: {'; '.join(page_errors)}"
                elif not pages_to_scan and all_warnings:
                    extraction_error = f"Corrupt or unreadable PDF content: {'; '.join(all_warnings)}"

            return ExtractedDocument(
                file_path=path,
                file_name=path.name,
                file_type="pdf",
                text_snippet=snippet,
                metadata=metadata,
                file_size_bytes=file_size,
                extraction_error=extraction_error,
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

    def _extract_rtf(self, path: Path, file_size: int) -> ExtractedDocument:
        """Extracts clean plain text from an RTF document by removing markup and control words."""
        raw_bytes = path.read_bytes()
        raw_text = ""
        for encoding in ("utf-8", "latin-1", "cp1252", "ascii"):
            try:
                raw_text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        pattern = re.compile(
            r"\\([a-z]{1,32})(-?\d+)? ?|\\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)",
            re.I,
        )
        destinations_to_skip = {
            "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer",
            "author", "operator", "generator", "keywords", "comment", "title", "subject",
        }

        stack: list[bool] = []
        ignorable = False
        just_opened_group = False
        ucskip = 1
        curskip = 0
        out: list[str] = []

        for match in pattern.finditer(raw_text):
            word, arg, hex_char, char, brace, raw_char = match.groups()
            if brace:
                curskip = 0
                if brace == "{":
                    stack.append(ignorable)
                    just_opened_group = True
                elif brace == "}":
                    just_opened_group = False
                    if stack:
                        ignorable = stack.pop()
            elif char:
                curskip = 0
                if char == "*" and just_opened_group:
                    ignorable = True
                elif not ignorable:
                    out.append(char)
                just_opened_group = False
            elif word:
                curskip = 0
                word_lower = word.lower()
                if word_lower in destinations_to_skip:
                    ignorable = True
                elif word_lower == "bin":
                    pass
                elif word_lower in ("par", "line"):
                    if not ignorable:
                        out.append("\n")
                elif word_lower == "tab":
                    if not ignorable:
                        out.append(" ")
                elif word_lower == "uc":
                    ucskip = int(arg) if arg else 1
                elif word_lower == "u":
                    c = int(arg) if arg else 0
                    if c < 0:
                        c += 0x10000
                    if not ignorable:
                        out.append(chr(c))
                    curskip = ucskip
                just_opened_group = False
            elif hex_char:
                just_opened_group = False
                if curskip > 0:
                    curskip -= 1
                elif not ignorable:
                    try:
                        c = bytes.fromhex(hex_char).decode("cp1252", errors="replace")
                        out.append(c)
                    except Exception:
                        pass
            elif raw_char:
                just_opened_group = False
                if curskip > 0:
                    curskip -= 1
                elif not ignorable:
                    out.append(raw_char)

        clean_text = "".join(out)
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n\s*\n+", "\n\n", clean_text)
        snippet = self._truncate(clean_text)

        return ExtractedDocument(
            file_path=path,
            file_name=path.name,
            file_type="rtf",
            text_snippet=snippet,
            file_size_bytes=file_size,
        )
