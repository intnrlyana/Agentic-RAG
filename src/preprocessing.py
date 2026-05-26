"""Lightweight text preprocessing for RAG-oriented page records."""

from __future__ import annotations

import re
from typing import Any


PageRecord = dict[str, Any]

_WHITESPACE_RE = re.compile(r"[ \t]+")
_REPEATED_NEWLINES_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
_SEPARATOR_LINE_RE = re.compile(r"^\s*[-_=]{3,}\s*$")


def clean_text(text: str) -> str:
    """Apply light normalization while preserving wording and structure."""
    if not text:
        return ""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")

    cleaned_lines: list[str] = []
    for line in lines:
        compact_line = _WHITESPACE_RE.sub(" ", line).strip()
        if not compact_line:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_LINE_RE.match(compact_line):
            continue
        if _SEPARATOR_LINE_RE.match(compact_line):
            continue
        cleaned_lines.append(compact_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = _REPEATED_NEWLINES_RE.sub("\n\n", cleaned_text)
    return cleaned_text.strip()


def preprocess_pages(pages: list[PageRecord], min_text_length: int = 30) -> list[PageRecord]:
    """Clean extracted pages while preserving page metadata and skip state."""
    processed_pages: list[PageRecord] = []

    for page in pages:
        original_text = page.get("text", "") or ""
        cleaned_text = clean_text(original_text)

        original_length = len(original_text)
        cleaned_length = len(cleaned_text)
        was_skipped, skip_reason = _determine_skip_state(cleaned_text, min_text_length=min_text_length)

        processed_page = {
            "source": page.get("source"),
            "page_number": page.get("page_number"),
            "source_type": page.get("source_type"),
            "text": cleaned_text,
            "original_text_length": original_length,
            "cleaned_text_length": cleaned_length,
            "was_skipped": was_skipped,
            "skip_reason": skip_reason,
        }
        processed_pages.append(processed_page)

    return processed_pages


def summarize_preprocessing(processed_pages: list[PageRecord]) -> dict[str, Any]:
    """Summarize retained and skipped pages after preprocessing."""
    retained_pages = [page for page in processed_pages if not page.get("was_skipped")]
    skipped_pages = [page for page in processed_pages if page.get("was_skipped")]

    skipped_by_reason: dict[str, int] = {}
    for page in skipped_pages:
        reason = page.get("skip_reason") or "unknown"
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    return {
        "total_pages": len(processed_pages),
        "retained_pages": len(retained_pages),
        "skipped_pages": len(skipped_pages),
        "total_original_characters": sum(page.get("original_text_length", 0) for page in processed_pages),
        "total_cleaned_characters": sum(page.get("cleaned_text_length", 0) for page in processed_pages),
        "skipped_by_reason": skipped_by_reason,
    }


def _determine_skip_state(cleaned_text: str, *, min_text_length: int) -> tuple[bool, str | None]:
    """Return skip state for empty or extremely short cleaned pages."""
    if not cleaned_text:
        return True, "empty_after_cleaning"
    if len(cleaned_text) < min_text_length:
        return True, "too_short_after_cleaning"
    return False, None
