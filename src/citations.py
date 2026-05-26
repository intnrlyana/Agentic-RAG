"""Citation helpers for presenting retrieved document references clearly."""

from __future__ import annotations

from typing import Any


RetrievedChunk = dict[str, Any]
CitationRecord = dict[str, Any]


def format_citations(retrieved_chunks: list[RetrievedChunk]) -> list[CitationRecord]:
    """Deduplicate and format citations from retrieved chunks in relevance order."""
    citations: list[CitationRecord] = []
    seen: set[tuple[str, int, str]] = set()

    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda chunk: (
            chunk.get("rank", 9999),
            -float(chunk.get("score", 0.0)),
        ),
    )

    for chunk in sorted_chunks:
        key = (chunk["source"], chunk["page_number"], chunk["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "citation_id": len(citations) + 1,
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "score": float(chunk.get("score", 0.0)),
                "rank": chunk.get("rank"),
                "chunk_text": chunk.get("chunk_text", ""),
            }
        )

    return citations


def render_citations_text(citations: list[CitationRecord]) -> str:
    """Render citations into a simple readable citation block."""
    if not citations:
        return ""

    lines = [
        f"[{citation['citation_id']}] {citation['source']}, Page {citation['page_number']}"
        for citation in citations
    ]
    return "\n".join(lines)


def attach_citations_to_response(answer: str, citations: list[CitationRecord]) -> dict[str, Any]:
    """Bundle a final answer with its formatted citations."""
    return {
        "answer": answer,
        "citations": citations,
        "citations_text": render_citations_text(citations),
    }
