"""Document loading utilities for uploaded and local PDF and DOCX sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader


logger = logging.getLogger(__name__)


PageRecord = dict[str, Any]


def extract_text_from_pdf_file(file: Any, source_type: str = "uploaded") -> list[PageRecord]:
    """Extract page text from a Streamlit-uploaded PDF-like object."""
    source_name = getattr(file, "name", "uploaded_document.pdf")

    try:
        if hasattr(file, "seek"):
            file.seek(0)
        reader = PdfReader(file)
    except Exception as exc:
        logger.warning("Failed to read uploaded PDF '%s': %s", source_name, exc)
        return []

    return _extract_pages_from_reader(reader, source_name=source_name, source_type=source_type)


def extract_text_from_pdf_path(pdf_path: str | Path, source_type: str = "local") -> list[PageRecord]:
    """Extract page text from a PDF file stored on disk."""
    path = Path(pdf_path)

    try:
        with path.open("rb") as pdf_file:
            reader = PdfReader(pdf_file)
            return _extract_pages_from_reader(
                reader,
                source_name=path.name,
                source_type=source_type,
            )
    except Exception as exc:
        logger.warning("Failed to read local PDF '%s': %s", path, exc)
        return []


def extract_text_from_docx_file(file: Any, source_type: str = "uploaded") -> list[PageRecord]:
    """Extract document text from a DOCX-like file object."""
    source_name = getattr(file, "name", "uploaded_document.docx")

    try:
        if hasattr(file, "seek"):
            file.seek(0)
        document = Document(file)
    except Exception as exc:
        logger.warning("Failed to read uploaded DOCX '%s': %s", source_name, exc)
        return []

    return _extract_pages_from_docx_document(
        document,
        source_name=source_name,
        source_type=source_type,
    )


def extract_text_from_docx_path(docx_path: str | Path, source_type: str = "local") -> list[PageRecord]:
    """Extract document text from a DOCX file stored on disk."""
    path = Path(docx_path)

    try:
        document = Document(path)
        return _extract_pages_from_docx_document(
            document,
            source_name=path.name,
            source_type=source_type,
        )
    except Exception as exc:
        logger.warning("Failed to read local DOCX '%s': %s", path, exc)
        return []


def load_supported_documents_from_data_folder(data_dir: str = "data") -> list[PageRecord]:
    """Load and extract text from all supported documents inside the local data folder."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Data directory does not exist: %s", data_path)
        return []

    document_paths = sorted(
        path
        for path in data_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
    )
    if not document_paths:
        logger.warning("No supported documents found in data directory: %s", data_path)
        return []

    extracted_pages: list[PageRecord] = []
    for document_path in document_paths:
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            extracted_pages.extend(extract_text_from_pdf_path(document_path, source_type="local"))
        elif suffix == ".docx":
            extracted_pages.extend(extract_text_from_docx_path(document_path, source_type="local"))

    return extracted_pages


def summarize_extraction(pages: list[PageRecord]) -> dict[str, Any]:
    """Summarize extracted page records for backend tracking and UI display."""
    source_filenames = sorted({page["source"] for page in pages})
    empty_pages = sum(1 for page in pages if not page.get("text", "").strip())

    pages_per_source: dict[str, int] = {}
    for page in pages:
        source = page["source"]
        pages_per_source[source] = pages_per_source.get(source, 0) + 1

    source_types = sorted({page["source_type"] for page in pages}) if pages else []

    return {
        "document_count": len(source_filenames),
        "total_pages_extracted": len(pages),
        "empty_or_skipped_pages": empty_pages,
        "source_filenames": source_filenames,
        "pages_per_source": pages_per_source,
        "source_types": source_types,
    }


def _extract_pages_from_reader(
    reader: PdfReader,
    *,
    source_name: str,
    source_type: str,
) -> list[PageRecord]:
    """Normalize page extraction and keep empty pages as explicit records."""
    extracted_pages: list[PageRecord] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            extracted_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Failed to extract page %s from '%s': %s",
                page_index,
                source_name,
                exc,
            )
            extracted_text = ""

        extracted_pages.append(
            {
                "source": source_name,
                "page_number": page_index,
                "text": extracted_text.strip(),
                "source_type": source_type,
            }
        )

    if not extracted_pages:
        logger.warning("PDF '%s' contains no readable pages.", source_name)

    return extracted_pages


def _extract_pages_from_docx_document(
    document: Document,
    *,
    source_name: str,
    source_type: str,
) -> list[PageRecord]:
    """Normalize DOCX extraction into a single page-like record."""
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    extracted_text = "\n\n".join(paragraphs)

    page_record = {
        "source": source_name,
        "page_number": 1,
        "text": extracted_text,
        "source_type": source_type,
    }

    if not extracted_text:
        logger.warning("DOCX '%s' contains no readable paragraphs.", source_name)

    return [page_record]
