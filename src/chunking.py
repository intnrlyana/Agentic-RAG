"""Chunking utilities for splitting preprocessed pages into retrieval-friendly segments."""

from __future__ import annotations

import re
from typing import Any


PageRecord = dict[str, Any]
ChunkRecord = dict[str, Any]

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Za-z]")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_HEADING_LINE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9/&(),\- ]{2,90}$")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list[str]:
    """Split text into overlapping chunks using paragraph-first assembly."""
    if not text:
        return []

    normalized_chunk_size = max(1, chunk_size)
    normalized_overlap = max(0, min(overlap, normalized_chunk_size - 1))
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) > normalized_chunk_size:
            if current_paragraphs:
                chunks.append(_join_paragraphs(current_paragraphs))
                current_paragraphs = []
            chunks.extend(
                _chunk_long_paragraph(
                    paragraph,
                    chunk_size=normalized_chunk_size,
                    overlap=normalized_overlap,
                )
            )
            continue

        if current_paragraphs and _joined_length(current_paragraphs + [paragraph]) > normalized_chunk_size:
            chunks.append(_join_paragraphs(current_paragraphs))
            current_paragraphs = _build_overlap_seed(current_paragraphs, normalized_overlap)
            while current_paragraphs and _joined_length(current_paragraphs + [paragraph]) > normalized_chunk_size:
                current_paragraphs.pop(0)

        current_paragraphs.append(paragraph)

    if current_paragraphs:
        chunks.append(_join_paragraphs(current_paragraphs))

    return _deduplicate_chunks(chunks)


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = 900,
    overlap: int = 180,
) -> list[ChunkRecord]:
    """Create chunk records for all retained preprocessed pages."""
    chunks: list[ChunkRecord] = []

    for page in pages:
        if page.get("was_skipped"):
            continue

        page_text = page.get("text", "") or ""
        text_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        page_headings = _extract_heading_candidates(page_text)

        for chunk_index, chunk_value in enumerate(text_chunks):
            is_low_information = _is_low_information_chunk(chunk_value)
            section_title, subsection_title = _resolve_chunk_headings(chunk_value, page_headings)
            chunk_record = {
                "chunk_id": f"{page['source']}_p{page['page_number']}_c{chunk_index}",
                "source": page.get("source"),
                "page_number": page.get("page_number"),
                "source_type": page.get("source_type"),
                "original_text_length": page.get("original_text_length"),
                "cleaned_text_length": page.get("cleaned_text_length"),
                "chunk_index": chunk_index,
                "chunk_text": chunk_value,
                "chunk_length": len(chunk_value),
                "is_low_information": is_low_information,
                "section_title": section_title,
                "subsection_title": subsection_title,
                "heading_path": " > ".join(
                    [value for value in [section_title, subsection_title] if value]
                ),
            }
            chunks.append(chunk_record)

    return chunks


def summarize_chunks(chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Summarize chunk output for debugging and UI display."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "chunks_with_headings": 0,
            "average_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "chunks_by_document": {},
        }

    chunk_lengths = [chunk["chunk_length"] for chunk in chunks]
    chunks_by_document: dict[str, int] = {}
    for chunk in chunks:
        source = chunk["source"]
        chunks_by_document[source] = chunks_by_document.get(source, 0) + 1

    return {
        "total_chunks": len(chunks),
        "total_documents": len(chunks_by_document),
        "chunks_with_headings": sum(1 for chunk in chunks if chunk.get("heading_path")),
        "average_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2),
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "chunks_by_document": chunks_by_document,
    }


def _adjust_start_boundary(text: str, start: int, lookahead: int = 40) -> int:
    """Move chunk starts away from mid-word positions when possible."""
    if start <= 0:
        return 0

    upper_bound = min(start + lookahead, len(text))
    for idx in range(start, upper_bound):
        if text[idx].isspace():
            return min(idx + 1, len(text))
    return start


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-like blocks while preserving local structure."""
    normalized_text = text.strip()
    if not normalized_text:
        return []

    raw_blocks = [block.strip() for block in _PARAGRAPH_BREAK_RE.split(normalized_text) if block.strip()]
    if raw_blocks:
        return raw_blocks

    return [line.strip() for line in normalized_text.splitlines() if line.strip()]


def _join_paragraphs(paragraphs: list[str]) -> str:
    """Join paragraph blocks into a chunk with visible paragraph boundaries."""
    return "\n\n".join(paragraphs).strip()


def _joined_length(paragraphs: list[str]) -> int:
    """Compute the joined chunk length for a paragraph list."""
    if not paragraphs:
        return 0
    return len(_join_paragraphs(paragraphs))


def _build_overlap_seed(paragraphs: list[str], overlap_chars: int) -> list[str]:
    """Reuse trailing paragraphs as overlap without cutting into new paragraphs."""
    if overlap_chars <= 0 or not paragraphs:
        return []

    seed: list[str] = []
    total_length = 0
    for paragraph in reversed(paragraphs):
        paragraph_length = len(paragraph)
        separator_length = 2 if seed else 0
        if seed and total_length + separator_length + paragraph_length > overlap_chars:
            break
        seed.insert(0, paragraph)
        total_length += separator_length + paragraph_length
        if total_length >= overlap_chars:
            break
    return seed


def _chunk_long_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback chunking for oversized paragraph blocks."""
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        adjusted_start = _adjust_start_boundary(text, start)
        ideal_end = min(adjusted_start + chunk_size, text_length)
        adjusted_end = _adjust_end_boundary(text, adjusted_start, ideal_end)
        chunk = text[adjusted_start:adjusted_end].strip()
        if chunk:
            chunks.append(chunk)
        if adjusted_end >= text_length:
            break
        next_start = max(adjusted_end - overlap, adjusted_start + 1)
        if next_start <= start:
            next_start = adjusted_end
        start = next_start

    return chunks


def _deduplicate_chunks(chunks: list[str]) -> list[str]:
    """Drop accidental duplicate chunks created by overlap seeding."""
    deduplicated: list[str] = []
    previous_chunk = ""
    for chunk in chunks:
        normalized_chunk = chunk.strip()
        if normalized_chunk and normalized_chunk != previous_chunk:
            deduplicated.append(normalized_chunk)
            previous_chunk = normalized_chunk
    return deduplicated


def _adjust_end_boundary(text: str, start: int, ideal_end: int, extension: int = 120) -> int:
    """Prefer sentence or whitespace boundaries near the target chunk end."""
    if ideal_end >= len(text):
        return len(text)

    forward_limit = min(len(text), ideal_end + extension)
    sentence_end = _find_sentence_boundary(text, ideal_end, forward_limit)
    if sentence_end is not None and sentence_end > start:
        return sentence_end

    whitespace_end = _find_whitespace_boundary(text, ideal_end, forward_limit)
    if whitespace_end is not None and whitespace_end > start:
        return whitespace_end

    backward_limit = max(start + 1, ideal_end - 80)
    for idx in range(ideal_end, backward_limit, -1):
        if text[idx - 1].isspace():
            return idx

    return ideal_end


def _find_sentence_boundary(text: str, start: int, end: int) -> int | None:
    """Find the next sentence boundary in a search window."""
    window = text[start:end]
    match = _SENTENCE_BOUNDARY_RE.search(window)
    if match is None:
        return None
    return start + match.end()


def _find_whitespace_boundary(text: str, start: int, end: int) -> int | None:
    """Find the next whitespace boundary in a search window."""
    for idx in range(start, end):
        if text[idx].isspace():
            return idx
    return None


def _is_low_information_chunk(text: str) -> bool:
    """Heuristically flag chunks that look like TOC/index material."""
    normalized_text = text.strip()
    if not normalized_text:
        return True

    lowered_text = normalized_text.lower()
    if "table of contents" in lowered_text or lowered_text.startswith("contents"):
        return True

    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    numbered_lines = sum(1 for line in lines if _NUMBERED_LINE_RE.match(line))
    short_lines = sum(1 for line in lines if len(line.split()) <= 8)
    punctuation_lines = sum(1 for line in lines if any(mark in line for mark in ".!?"))

    numbered_ratio = numbered_lines / len(lines)
    short_ratio = short_lines / len(lines)
    punctuation_ratio = punctuation_lines / len(lines)

    return numbered_ratio >= 0.5 and short_ratio >= 0.7 and punctuation_ratio <= 0.6


def _extract_heading_candidates(text: str) -> list[str]:
    """Find heading-like lines that can be attached as chunk metadata."""
    headings: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) > 90:
            continue
        if _looks_like_heading(candidate):
            headings.append(candidate)
    return headings


def _resolve_chunk_headings(chunk_text: str, page_headings: list[str]) -> tuple[str | None, str | None]:
    """Resolve the best section and subsection heading for a chunk."""
    chunk_headings = _extract_heading_candidates(chunk_text)
    headings = chunk_headings or page_headings
    if not headings:
        return None, None

    section_title = headings[0]
    subsection_title = headings[1] if len(headings) > 1 else None
    return section_title, subsection_title


def _looks_like_heading(line: str) -> bool:
    """Heuristically classify document lines as section headings."""
    if not _HEADING_LINE_RE.match(line):
        return False
    if line.endswith((".", ";", ":")):
        return False

    words = line.split()
    if len(words) < 2 or len(words) > 14:
        return False

    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    titleish_words = sum(1 for word in words if word[:1].isupper())
    lowercase_words = sum(1 for word in words if word[:1].islower())

    if uppercase_words >= max(1, len(words) // 2):
        return True
    if titleish_words >= max(2, len(words) - 1) and lowercase_words <= 2:
        return True
    if _NUMBERED_LINE_RE.match(line):
        return True
    return False
