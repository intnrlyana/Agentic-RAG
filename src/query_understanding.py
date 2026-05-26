"""Query understanding helpers for normalization and lightweight query splitting."""

from __future__ import annotations

import re
from typing import Any


_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def should_rewrite_query(
    user_query: str,
    retrieval_evaluation: dict[str, Any] | None = None,
) -> bool:
    """Only rewrite short, vague, or weakly retrieved questions."""
    normalized_query = normalize_query(user_query)
    if not normalized_query:
        return False

    query_tokens = normalized_query.split()
    vague_terms = {
        "this",
        "that",
        "it",
        "they",
        "them",
        "thing",
        "things",
        "something",
        "stuff",
        "policy",
        "document",
        "manual",
        "report",
    }

    if len(query_tokens) < 6:
        return True
    if any(token in vague_terms for token in query_tokens):
        return True

    if retrieval_evaluation:
        if not retrieval_evaluation.get("is_relevant", False):
            return True
        if retrieval_evaluation.get("average_keyword_overlap", 0.0) < 0.05:
            return True

    return False


def rewrite_query(user_query: str) -> str:
    """Rewrite a vague query into a concise retrieval-friendly phrase locally."""
    normalized_query = normalize_query(user_query)
    if not normalized_query:
        return user_query.strip()
    return normalized_query


def analyze_query_intent(user_query: str) -> dict[str, Any]:
    """Return only lightweight normalized query metadata."""
    normalized_query = normalize_query(user_query)
    tokens = normalized_query.split()
    return {
        "normalized_query": normalized_query,
        "tokens": tokens,
    }


def extract_query_topics(user_query: str) -> list[set[str]]:
    """Split a compound query into topic term groups."""
    normalized_text = user_query.lower().strip()
    normalized_text = re.sub(r"[?!.:,;/]+", " ", normalized_text)
    raw_parts = re.split(r"\b(?:or|and|across|versus|vs)\b", normalized_text)
    topics: list[set[str]] = []
    for part in raw_parts:
        part_terms = support_terms(part)
        if part_terms:
            topics.append(part_terms)

    deduped_topics: list[set[str]] = []
    for topic in topics:
        if any(topic == existing for existing in deduped_topics):
            continue
        deduped_topics.append(topic)
    return deduped_topics[:4]


def split_coordinated_question(user_query: str) -> list[str]:
    """Split a coordinated question into item-specific subquestions while preserving shared framing."""
    normalized = " ".join(user_query.strip().split())
    if not normalized:
        return []

    match = re.match(
        r"^(?P<prefix>.+?\b)(?P<item1>[A-Za-z0-9][^?.,;:]*?)\s+(?:and|or)\s+(?P<item2>[A-Za-z0-9][^?.,;:]*?)(?P<suffix>\s+(?:to|for|at|in|on|during|under)\b.*)?[?!.]?\s*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return []

    prefix = " ".join((match.group("prefix") or "").split())
    item1 = " ".join((match.group("item1") or "").split())
    item2 = " ".join((match.group("item2") or "").split())
    suffix = " ".join((match.group("suffix") or "").split())

    if not prefix or not item1 or not item2:
        return []

    questions = [
        " ".join(part for part in [prefix, item1, suffix] if part).strip(),
        " ".join(part for part in [prefix, item2, suffix] if part).strip(),
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for question in questions:
        lowered = question.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(question)
    return deduped[:4]


def build_query_variants(
    original_query: str,
    active_query: str,
    *,
    canonical_query: str | None = None,
    rewrite_query_text: str | None = None,
) -> list[str]:
    """Create a small set of retrieval-oriented query variants."""
    expanded_query = expand_retry_query(original_query, active_query)
    normalized_original = normalize_query(original_query)
    normalized_active = normalize_query(active_query)
    if normalized_original and normalized_original == normalized_active:
        expanded_query = normalized_original

    variants = [
        original_query.strip(),
        active_query.strip(),
        (canonical_query or "").strip(),
        normalized_original,
        normalized_active,
        expanded_query,
    ]
    if rewrite_query_text and rewrite_query_text.strip():
        variants.append(rewrite_query_text.strip())

    deduplicated: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = _MULTISPACE_RE.sub(" ", variant.strip())
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduplicated.append(normalized)
    return deduplicated[:6]


def normalize_query(user_query: str) -> str:
    """Normalize a user query into a compact retrieval phrase."""
    lowered = user_query.lower().strip()
    lowered = _NON_WORD_RE.sub(" ", lowered)
    lowered = _MULTISPACE_RE.sub(" ", lowered)
    filler_terms = {
        "a",
        "an",
        "any",
        "and",
        "about",
        "are",
        "can",
        "could",
        "do",
        "does",
        "for",
        "get",
        "give",
        "how",
        "i",
        "is",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "the",
        "there",
        "to",
        "we",
        "what",
        "will",
        "would",
    }
    tokens = [token for token in lowered.split() if token not in filler_terms]
    return " ".join(tokens)


def support_terms(text: str) -> set[str]:
    """Tokenize text for coarse grounding and matching checks."""
    normalized = normalize_query(text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


def expand_retry_query(original_query: str, current_query: str) -> str:
    """Deterministically broaden the query for one retry attempt."""
    normalized_original = normalize_query(original_query)
    current_tokens = current_query.strip()
    if normalized_original and current_tokens:
        return f"{normalized_original} {current_tokens}"
    return normalized_original or current_tokens or original_query.strip()
