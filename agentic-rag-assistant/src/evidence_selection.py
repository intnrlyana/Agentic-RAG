"""Evidence selection helpers for answer-time chunk narrowing and extractive rescue."""

from __future__ import annotations

import re
from typing import Any

from src.query_understanding import analyze_query_intent, support_terms


RetrievedChunk = dict[str, Any]
_TEXT_SEGMENT_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")
_FAQ_QUESTION_RE = re.compile(r"^\s*(\d+)[.)]?\s+.+\?$")
_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred",
}
_POSITIVE_FACT_CUES = {
    "entitled", "includes", "include", "consists", "available", "eligible",
    "maximum", "minimum", "within", "up", "up to", "between", "period",
}
_DURATION_CUES = {
    "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
}


def build_extractive_rescue_answer(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    query_intent: dict[str, Any] | None = None,
) -> str | None:
    """Extract a direct answer sentence from the strongest retrieved chunks."""
    query_intent = query_intent or analyze_query_intent(user_query)
    query_terms = support_terms(user_query)
    if not query_terms or not retrieved_chunks:
        return None

    if query_intent.get("expects_quantity") or query_intent.get("expects_duration"):
        structured_answer = build_structured_fact_answer(
            user_query,
            retrieved_chunks,
            query_intent=query_intent,
        )
        if structured_answer:
            return structured_answer

    if query_intent.get("is_compound"):
        policy_answer = build_multi_topic_extractive_answer(
            query_intent,
            retrieved_chunks,
        )
        if policy_answer:
            return policy_answer

    faq_answer = extract_faq_answer_block(user_query, retrieved_chunks[:3])
    if faq_answer:
        return faq_answer

    best_segment = ""
    best_score = 0.0

    for chunk_index, chunk in enumerate(retrieved_chunks[:3]):
        chunk_strength = max(
            float(chunk.get("keyword_overlap", 0.0)),
            float(chunk.get("score", 0.0)),
            float(chunk.get("reranker_score", float("-inf"))) if chunk.get("reranker_score") is not None else 0.0,
        )
        if chunk_strength < 0.35:
            continue

        segments = split_text_segments(chunk.get("chunk_text", ""))
        for segment_index, segment in enumerate(segments):
            segment_terms = support_terms(segment)
            if not segment_terms:
                continue

            overlap = len(query_terms & segment_terms) / max(1, len(query_terms))
            faq_boost = 0.0
            is_question_segment = segment.strip().endswith("?") or "?" in segment
            if not is_question_segment and len(segment.split()) >= 5:
                faq_boost += 0.12
            if any(
                marker in segment.lower()
                for marker in ("yes", "no", "not", "provided", "included", "available", "entitled", "eligible")
            ):
                faq_boost += 0.12

            previous_segment = segments[segment_index - 1] if segment_index > 0 else ""
            if previous_segment.endswith("?"):
                previous_terms = support_terms(previous_segment)
                question_overlap = len(query_terms & previous_terms) / max(1, len(query_terms))
                faq_boost += 0.2 * question_overlap

            next_segment = segments[segment_index + 1] if segment_index + 1 < len(segments) else ""
            if is_question_segment:
                faq_boost -= 0.2
                if next_segment and not next_segment.strip().endswith("?"):
                    next_terms = support_terms(next_segment)
                    next_overlap = len(query_terms & next_terms) / max(1, len(query_terms & segment_terms))
                    if next_overlap >= 0.2 or overlap >= 0.4:
                        answer_score = overlap + faq_boost + 0.25 + max(0.0, next_overlap)
                        if answer_score > best_score:
                            best_score = answer_score
                            best_segment = next_segment.strip()
                        continue

            rank_bonus = max(0.0, 0.08 - (chunk_index * 0.02))
            score = overlap + faq_boost + rank_bonus
            if score > best_score:
                best_score = score
                best_segment = segment.strip()

    if best_score < 0.45 or not best_segment or best_segment.endswith("?"):
        return None
    return best_segment


def build_structured_fact_answer(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    query_intent: dict[str, Any] | None = None,
) -> str | None:
    """Prefer numeric/entitlement/duration answer lines for structured fact questions."""
    query_intent = query_intent or analyze_query_intent(user_query)
    query_terms = support_terms(user_query)
    best_segment = ""
    best_score = 0.0

    for chunk_index, chunk in enumerate(retrieved_chunks[:3]):
        structured_lines = prepare_structured_lines(chunk.get("chunk_text", ""))
        last_heading = ""
        for line_index, segment in enumerate(structured_lines):
            if not segment:
                continue
            if _looks_like_heading_line(segment):
                last_heading = segment
                continue
            if segment.endswith("?"):
                continue

            segment_terms = support_terms(segment)
            if not segment_terms:
                continue

            overlap = len(query_terms & segment_terms) / max(1, len(query_terms))
            has_digits = bool(_DIGIT_RE.search(segment))
            has_spelled_number = any(word in segment.lower().split() for word in _NUMBER_WORDS)
            entitlement_boost = 0.0
            lowered_segment = segment.lower()
            if any(cue in lowered_segment for cue in _POSITIVE_FACT_CUES):
                entitlement_boost += 0.2
            if query_intent.get("expects_duration") and any(cue in lowered_segment for cue in _DURATION_CUES):
                entitlement_boost += 0.2
            if has_digits or has_spelled_number:
                entitlement_boost += 0.2

            heading_terms = support_terms(last_heading)
            heading_overlap = len(query_terms & heading_terms) / max(1, len(query_terms)) if heading_terms else 0.0
            if heading_overlap >= 0.4 and (has_digits or has_spelled_number):
                entitlement_boost += 0.25
            if heading_overlap >= 0.4 and any(cue in lowered_segment for cue in _POSITIVE_FACT_CUES | _DURATION_CUES):
                entitlement_boost += 0.2

            penalty = 0.0
            if re.search(r"\b(?:shall|will|may)\s+not\b", lowered_segment):
                penalty += 0.35
            if re.search(r"\bwithout\b|\bunless\b|\bexcept\b|\bsubject to\b", lowered_segment):
                penalty += 0.2
            if re.search(r"\bif\b", lowered_segment) and not (has_digits or has_spelled_number):
                penalty += 0.15
            if "provided that" in lowered_segment:
                penalty += 0.15

            proximity_boost = max(0.0, 0.08 - (chunk_index * 0.02) - (line_index * 0.01))
            score = overlap + entitlement_boost + proximity_boost - penalty
            if score > best_score:
                best_score = score
                best_segment = segment.strip()

    if best_score < 0.45 or not best_segment:
        return None
    return best_segment


def select_chunks_for_grounded_interpretation(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_chunks: int = 2,
) -> list[RetrievedChunk]:
    """Select a very small, high-signal context set for policy interpretation questions."""
    if not retrieved_chunks:
        return []

    query_terms = support_terms(user_query)
    scored_chunks: list[tuple[float, RetrievedChunk]] = []
    for chunk in retrieved_chunks[:5]:
        chunk_text = chunk.get("chunk_text", "")
        chunk_terms = support_terms(chunk_text)
        overlap = len(query_terms & chunk_terms) / max(1, len(query_terms)) if query_terms else 0.0
        heading_boost = 0.0
        first_lines = " ".join(prepare_structured_lines(chunk_text)[:3]).lower()
        if query_terms and any(term in first_lines for term in query_terms):
            heading_boost = 0.12
        score = (
            overlap
            + heading_boost
            + (0.25 * float(chunk.get("keyword_overlap", 0.0)))
            + min(0.15, float(chunk.get("score", 0.0)) * 0.2)
        )
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored_chunks[: max(1, max_chunks)]]
    return selected or retrieved_chunks[: max(1, max_chunks)]


def _looks_like_heading_line(line: str) -> bool:
    """Detect short section-heading lines in a structured block."""
    lowered = line.lower().strip()
    if not lowered:
        return False
    if lowered.endswith(":"):
        return True
    if len(lowered.split()) <= 6 and not _DIGIT_RE.search(lowered):
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\.?\s+[a-z].{0,80}$", lowered))


def extract_faq_answer_block(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str | None:
    """Prefer a structured FAQ answer block when the chunk contains multiple Q&A items."""
    query_terms = support_terms(user_query)
    best_answer = ""
    best_score = 0.0

    for chunk_index, chunk in enumerate(retrieved_chunks):
        lines = prepare_structured_lines(chunk.get("chunk_text", ""))
        if not lines:
            continue

        faq_items = extract_faq_items(lines)
        for item in faq_items:
            question_terms = support_terms(item["question"])
            if not question_terms:
                continue

            question_overlap = len(query_terms & question_terms) / max(1, len(query_terms))
            answer_text = " ".join(item["answer_lines"]).strip()
            answer_terms = support_terms(answer_text)
            answer_overlap = len(query_terms & answer_terms) / max(1, len(query_terms)) if answer_terms else 0.0
            rank_bonus = max(0.0, 0.1 - (chunk_index * 0.03))
            score = question_overlap + (0.35 * answer_overlap) + rank_bonus

            if score > best_score and answer_text:
                best_score = score
                best_answer = answer_text

    if best_score < 0.45:
        return None
    return best_answer or None


def select_context_chunks_for_answer(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_context_chunks: int,
    query_intent: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Keep only the strongest topic-matching chunks for answer generation."""
    if not retrieved_chunks:
        return []

    query_intent = query_intent or analyze_query_intent(user_query)
    selected_limit = max(1, max_context_chunks)
    query_terms = support_terms(user_query)
    if not query_terms:
        return retrieved_chunks[:selected_limit]

    scored_chunks: list[tuple[float, RetrievedChunk]] = []
    for chunk in retrieved_chunks:
        chunk_terms = support_terms(chunk.get("chunk_text", ""))
        if not chunk_terms:
            continue

        term_overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
        keyword_overlap = float(chunk.get("keyword_overlap", 0.0))
        reranker_score = float(chunk.get("reranker_score", float("-inf"))) if chunk.get("reranker_score") is not None else float("-inf")
        reranker_bonus = 0.12 if reranker_score >= 0.0 else 0.0
        semantic_bonus = min(0.15, float(chunk.get("score", 0.0)) * 0.2)
        selection_score = term_overlap + (0.35 * keyword_overlap) + reranker_bonus + semantic_bonus
        scored_chunks.append((selection_score, chunk))

    if not scored_chunks:
        return retrieved_chunks[:selected_limit]

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    best_score = scored_chunks[0][0]
    top_chunk = scored_chunks[0][1]
    top_chunk_overlap = len(query_terms & support_terms(top_chunk.get("chunk_text", ""))) / max(1, len(query_terms))

    if query_intent.get("is_compound"):
        selected_chunks = select_chunks_for_compound_query(
            scored_chunks,
            query_intent,
            selected_limit=min(max(2, selected_limit), 4),
        )
        if selected_chunks:
            return selected_chunks

    if query_intent.get("is_direct_fact") and top_chunk_overlap >= 0.5:
        direct_chunks = [
            chunk
            for score, chunk in scored_chunks
            if score >= max(0.55, best_score - 0.12)
        ]
        return direct_chunks[: min(2, selected_limit)]

    filtered_chunks = [
        chunk
        for score, chunk in scored_chunks
        if score >= max(0.45, best_score - 0.2)
    ]
    if filtered_chunks:
        return filtered_chunks[:selected_limit]
    return [top_chunk]


def split_text_segments(text: str) -> list[str]:
    """Split chunk text into compact answer-sized segments."""
    raw_segments = [
        segment.strip(" -\t")
        for segment in _TEXT_SEGMENT_SPLIT_RE.split(text.strip())
        if segment.strip(" -\t")
    ]
    if not raw_segments:
        return []

    merged_segments: list[str] = []
    current_segment = raw_segments[0]
    for next_segment in raw_segments[1:]:
        if should_merge_wrapped_segment(current_segment, next_segment):
            current_segment = f"{current_segment} {next_segment}".strip()
            continue
        merged_segments.append(current_segment)
        current_segment = next_segment

    merged_segments.append(current_segment)
    return merged_segments


def prepare_structured_lines(text: str) -> list[str]:
    """Normalize wrapped PDF lines while preserving FAQ question boundaries."""
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return []

    prepared_lines: list[str] = []
    current_line = raw_lines[0]
    for next_line in raw_lines[1:]:
        if is_faq_question_line(next_line):
            prepared_lines.append(current_line.strip())
            current_line = next_line
            continue
        if should_merge_wrapped_segment(current_line, next_line):
            current_line = f"{current_line} {next_line}".strip()
            continue
        prepared_lines.append(current_line.strip())
        current_line = next_line

    prepared_lines.append(current_line.strip())
    return prepared_lines


def extract_faq_items(lines: list[str]) -> list[dict[str, Any]]:
    """Extract question-and-answer blocks from structured FAQ-like lines."""
    items: list[dict[str, Any]] = []
    current_item: dict[str, Any] | None = None

    for line in lines:
        if is_faq_question_line(line):
            if current_item and current_item["answer_lines"]:
                items.append(current_item)
            current_item = {"question": line, "answer_lines": []}
            continue

        if current_item is not None:
            current_item["answer_lines"].append(line)

    if current_item and current_item["answer_lines"]:
        items.append(current_item)
    return items


def is_faq_question_line(line: str) -> bool:
    """Identify FAQ-style question headings within a chunk."""
    normalized_line = line.strip()
    return bool(normalized_line.endswith("?") and _FAQ_QUESTION_RE.match(normalized_line))


def should_merge_wrapped_segment(current_segment: str, next_segment: str) -> bool:
    """Merge line-wrapped PDF fragments into a fuller sentence."""
    if not current_segment or not next_segment:
        return False
    if current_segment.endswith((".", "!", "?", ":")):
        return False
    if next_segment[0].isdigit():
        return False
    trailing_token = current_segment.rstrip().split()[-1].lower().strip("(),")
    if trailing_token in {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "than",
        "the",
        "to",
        "with",
    }:
        return True
    if next_segment[:1].islower():
        return True
    if len(current_segment.split()) >= 18:
        return False
    return True


def select_chunks_for_compound_query(
    scored_chunks: list[tuple[float, RetrievedChunk]],
    query_intent: dict[str, Any],
    *,
    selected_limit: int,
) -> list[RetrievedChunk]:
    """Greedily select chunks that together cover multiple query topics."""
    topics = query_intent.get("topics", [])
    if not topics:
        return [chunk for _, chunk in scored_chunks[:selected_limit]]

    selected_chunks: list[RetrievedChunk] = []
    for topic_terms in topics:
        best_chunk: RetrievedChunk | None = None
        best_topic_score = 0.0
        for base_score, chunk in scored_chunks:
            chunk_terms = support_terms(chunk.get("chunk_text", ""))
            topic_overlap = len(topic_terms & chunk_terms) / max(1, len(topic_terms))
            if topic_overlap <= 0:
                continue
            topic_score = base_score + topic_overlap
            if topic_score > best_topic_score:
                best_topic_score = topic_score
                best_chunk = chunk

        if best_chunk is not None and best_chunk not in selected_chunks:
            selected_chunks.append(best_chunk)
            if len(selected_chunks) >= selected_limit:
                return selected_chunks[:selected_limit]

    for _, chunk in scored_chunks:
        if chunk in selected_chunks:
            continue
        selected_chunks.append(chunk)
        if len(selected_chunks) >= selected_limit:
            break

    return selected_chunks[:selected_limit]


def build_multi_topic_extractive_answer(
    query_intent: dict[str, Any],
    retrieved_chunks: list[RetrievedChunk],
) -> str | None:
    """Assemble a concise answer that covers each detected topic with supported evidence."""
    topics = query_intent.get("topics", [])
    if len(topics) < 2:
        return None

    topic_answers: list[str] = []
    for topic_terms in topics:
        best_segment = ""
        best_score = 0.0
        for chunk in retrieved_chunks[:4]:
            for segment in split_text_segments(chunk.get("chunk_text", "")):
                segment_terms = support_terms(segment)
                if not segment_terms or segment.endswith("?"):
                    continue
                overlap = len(topic_terms & segment_terms) / max(1, len(topic_terms))
                if overlap > best_score:
                    best_score = overlap
                    best_segment = segment.strip()
        if best_segment and best_score >= 0.3:
            topic_answers.append(best_segment)

    if not topic_answers:
        return None
    return " ".join(dict.fromkeys(topic_answers))
