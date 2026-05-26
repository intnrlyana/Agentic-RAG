"""Answer validation helpers for post-generation checks and grounding support."""

from __future__ import annotations

import re
from typing import Any

from src.query_understanding import analyze_query_intent, support_terms


RetrievedChunk = dict[str, Any]
_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def extract_answer_claims(answer: str) -> list[str]:
    """Split a generated answer into simple claim units for grounding checks."""
    claims = [
        segment.strip(" -\t")
        for segment in _CLAIM_SPLIT_RE.split(answer.strip())
        if segment.strip(" -\t")
    ]
    return claims or [answer.strip()]


def claim_has_support(claim: str, retrieved_chunks: list[RetrievedChunk]) -> bool:
    """Check whether a claim is sufficiently supported by any retrieved chunk."""
    claim_terms = support_terms(claim)
    if not claim_terms:
        return True

    for chunk in retrieved_chunks:
        chunk_text = chunk.get("chunk_text", "")
        chunk_terms = support_terms(chunk_text)
        if not chunk_terms:
            continue

        overlap_ratio = len(claim_terms & chunk_terms) / len(claim_terms)
        if overlap_ratio >= 0.4:
            return True

        claim_text = claim.strip().lower()
        if len(claim_text) >= 20 and claim_text[:20] in chunk_text.lower():
            return True

    return False


def answer_is_off_topic(
    user_query: str,
    answer: str,
    context_chunks: list[RetrievedChunk],
    *,
    query_intent: dict[str, Any] | None = None,
    fallback_answer: str,
) -> bool:
    """Reject answers that are weakly grounded or drift away from the narrowed context."""
    return assess_answer_quality(
        user_query,
        answer,
        context_chunks,
        query_intent=query_intent,
        fallback_answer=fallback_answer,
    )["off_topic"]


def assess_answer_quality(
    user_query: str,
    answer: str,
    context_chunks: list[RetrievedChunk],
    *,
    query_intent: dict[str, Any] | None = None,
    fallback_answer: str,
) -> dict[str, Any]:
    """Return a structured explanation for whether an answer is trusted."""
    if not answer.strip() or answer == fallback_answer or not context_chunks:
        return {
            "off_topic": False,
            "reason": "Answer was empty, fallback, or no context was available.",
            "unsupported_claims": [],
            "query_overlap": None,
            "top_chunk_overlap": None,
        }

    query_intent = query_intent or analyze_query_intent(user_query)
    claims = extract_answer_claims(answer)
    unsupported_claims = [
        claim for claim in claims
        if not claim_has_support(claim, context_chunks)
    ]
    if len(unsupported_claims) >= max(1, len(claims) // 2):
        return {
            "off_topic": True,
            "reason": "At least half of the generated claims did not have support in the selected context.",
            "unsupported_claims": unsupported_claims,
            "query_overlap": None,
            "top_chunk_overlap": None,
        }

    query_terms = support_terms(user_query)
    answer_terms = support_terms(answer)
    top_chunk_terms = support_terms(context_chunks[0].get("chunk_text", ""))
    query_overlap = len(query_terms & answer_terms) / max(1, len(query_terms)) if query_terms else 1.0
    top_chunk_overlap = len(answer_terms & top_chunk_terms) / max(1, len(answer_terms)) if answer_terms else 1.0
    if _misses_salient_query_terms(user_query, answer, context_chunks):
        return {
            "off_topic": True,
            "reason": "The answer missed the user's salient topic terms even though the top chunk contained them.",
            "unsupported_claims": unsupported_claims,
            "query_overlap": round(query_overlap, 4),
            "top_chunk_overlap": round(top_chunk_overlap, 4),
        }
    if (query_intent.get("expects_quantity") or query_intent.get("expects_duration")) and not _answer_has_structured_fact_signal(answer):
        return {
            "off_topic": True,
            "reason": "The question expects a quantity or duration, but the answer did not contain a structured fact signal.",
            "unsupported_claims": unsupported_claims,
            "query_overlap": round(query_overlap, 4),
            "top_chunk_overlap": round(top_chunk_overlap, 4),
        }
    if query_intent.get("is_compound") and not answer_covers_compound_topics(answer, query_intent):
        return {
            "off_topic": True,
            "reason": "The answer did not cover enough of the compound query topics.",
            "unsupported_claims": unsupported_claims,
            "query_overlap": round(query_overlap, 4),
            "top_chunk_overlap": round(top_chunk_overlap, 4),
        }
    off_topic = query_overlap < 0.35 or top_chunk_overlap < 0.25
    return {
        "off_topic": off_topic,
        "reason": (
            "The answer had weak lexical overlap with the question or the top chunk."
            if off_topic
            else "The generated answer stayed on topic and remained sufficiently grounded."
        ),
        "unsupported_claims": unsupported_claims,
        "query_overlap": round(query_overlap, 4),
        "top_chunk_overlap": round(top_chunk_overlap, 4),
    }


def answer_covers_compound_topics(answer: str, query_intent: dict[str, Any]) -> bool:
    """Require a compound-question answer to cover most detected topic groups."""
    answer_terms = support_terms(answer)
    topics = query_intent.get("topics", [])
    if not answer_terms or not topics:
        return True

    covered = 0
    for topic_terms in topics:
        overlap = len(topic_terms & answer_terms) / max(1, len(topic_terms))
        if overlap >= 0.3:
            covered += 1
    return covered >= max(1, min(len(topics), 2))


def _answer_has_structured_fact_signal(answer: str) -> bool:
    """Require quantity/duration answers to actually contain a structured fact."""
    lowered_answer = answer.lower()
    if _DIGIT_RE.search(answer):
        return True
    return any(
        phrase in lowered_answer
        for phrase in (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "fifteen",
            "sixteen",
            "twenty",
            "working days",
            "calendar year",
            "entitled to",
            "maximum",
            "minimum",
        )
    )


def _misses_salient_query_terms(
    user_query: str,
    answer: str,
    context_chunks: list[RetrievedChunk],
) -> bool:
    """Reject answers that ignore the distinctive terms from the user's query."""
    query_terms = support_terms(user_query)
    salient_terms = {
        term for term in query_terms
        if len(term) >= 4
    }
    if not salient_terms:
        return False

    answer_terms = support_terms(answer)
    if salient_terms & answer_terms:
        return False

    top_chunk_terms = support_terms(context_chunks[0].get("chunk_text", "")) if context_chunks else set()
    # If the answer misses the salient term but the top chunk clearly contains it, treat this as drift.
    return bool(salient_terms & top_chunk_terms)
