"""Basic and lightweight agentic RAG helpers using a local Ollama model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

from src.answer_validation import claim_has_support, extract_answer_claims
from src.citations import attach_citations_to_response, format_citations
from src.evidence_selection import prepare_structured_lines
from src.query_understanding import (
    analyze_query_intent,
    build_query_variants,
    normalize_query,
    rewrite_query,
    should_rewrite_query,
    support_terms,
)
from src.vector_store import calculate_keyword_overlap, rerank_chunks, search_similar_chunks


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
if not (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env.example", override=False)

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if GROQ_API_KEY else "ollama").strip().lower()
DEFAULT_OLLAMA_MODEL = (
    os.getenv("GROQ_MODEL_NAME", os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile"))
    if LLM_PROVIDER == "groq"
    else os.getenv("OLLAMA_MODEL_NAME", os.getenv("LLM_MODEL_NAME", "llama3.2:3b"))
)
OLLAMA_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
OLLAMA_READ_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS", "120"))
ENABLE_LLM_ANSWER_REPAIR = os.getenv("ENABLE_LLM_ANSWER_REPAIR", "false").lower() == "true"
STRICT_GROUNDING_FALLBACK = os.getenv("STRICT_GROUNDING_FALLBACK", "false").lower() == "true"
FALLBACK_ANSWER = "I could not find this information in the uploaded documents."

RetrievedChunk = dict[str, Any]
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class OllamaServiceError(RuntimeError):
    """Raised when Ollama is unavailable or does not finish in time."""

    def __init__(self, message: str, *, status_code: int = 504) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def build_context_from_chunks(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Combine retrieved chunks into a structured context block for the LLM."""
    return build_context_from_chunks_limited(retrieved_chunks)


def build_context_from_chunks_limited(
    retrieved_chunks: list[RetrievedChunk],
    max_context_chunks: int = 4,
    max_chars_per_chunk: int = 1200,
) -> str:
    """Combine a limited number of retrieved chunks into a structured context block."""
    if not retrieved_chunks:
        return ""

    selected_chunks = retrieved_chunks[: max(1, max_context_chunks)]
    context_blocks = []
    for chunk in selected_chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"Source: {chunk['source']}",
                    f"Page: {chunk['page_number']}",
                    f"Chunk ID: {chunk['chunk_id']}",
                    "Content:",
                    (chunk["chunk_text"] or "")[: max(1, max_chars_per_chunk)].strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def build_basic_rag_prompt(
    user_query: str,
    context: str,
    retrieval_query: str | None = None,
    primary_evidence: str | None = None,
    evidence_mode: str = "interpretive",
) -> str:
    """Create a grounded prompt that allows conservative interpretation from retrieved context."""
    query_block = user_query
    if retrieval_query and retrieval_query.strip() and retrieval_query.strip() != user_query.strip():
        query_block = (
            f"Original user question: {user_query.strip()}\n"
            f"Retrieval interpretation: {retrieval_query.strip()}"
        )

    evidence_guidance = """If the primary evidence directly answers the question, answer directly and confidently.
Do not add extra caution when the answer is already stated in the evidence."""
    if evidence_mode != "explicit":
        evidence_guidance = """If the primary evidence does not directly answer the question but the retrieved text clearly supports a careful interpretation,
answer naturally from the evidence first and add only a brief qualification if it is truly needed."""

    primary_evidence_block = ""
    if primary_evidence and primary_evidence.strip():
        primary_evidence_block = (
            "\nPrimary evidence (use this first; use the full context only as support):\n"
            f"{primary_evidence.strip()}\n"
        )

    return f"""You are a document-grounded assistant.
Use only the provided context to answer the user's question.
Do not use outside knowledge.
Answer in English.

{evidence_guidance}

If the user's wording does not exactly match the document wording but the policy or text clearly answers the question in substance,
give a concise grounded paraphrase.
When some qualification is needed:
- keep it brief and natural
- base it only on the retrieved text
- avoid repetitive phrases such as "the exact wording is not explicitly stated" unless that detail is essential
- do not invent extra rules, permissions, exceptions, or facts

Only use the fallback answer if the retrieved context is genuinely too weak to support either:
1. a direct answer, or
2. a careful interpretation from the text.

Fallback answer:
{FALLBACK_ANSWER}

Keep the answer concise, factual, and grounded.
If the retrieval interpretation is more specific than the original wording, use it to understand the user's intent.

User question:
{query_block}
{primary_evidence_block}

Context:
{context}
"""


def select_primary_evidence(
    user_query: str,
    top_chunk: RetrievedChunk | None,
) -> dict[str, Any]:
    """Extract the strongest answer-like lines from the top chunk and classify explicitness."""
    if not top_chunk:
        return {"text": "", "lines": [], "mode": "interpretive", "score": 0.0}

    query_terms = support_terms(user_query)
    lines = prepare_structured_lines(top_chunk.get("chunk_text", ""))
    if not lines:
        return {"text": "", "lines": [], "mode": "interpretive", "score": 0.0}

    scored_lines: list[dict[str, Any]] = []
    previous_line = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.endswith("?"):
            previous_line = stripped
            continue

        line_terms = support_terms(stripped)
        if not line_terms:
            previous_line = stripped
            continue

        overlap = len(query_terms & line_terms) / max(1, len(query_terms)) if query_terms else 0.0
        lexical_bonus = 0.0
        lowered = stripped.lower()
        if re.search(r"\b\d+(?:\.\d+)?\b", lowered):
            lexical_bonus += 0.12
        if any(
            cue in lowered
            for cue in ("entitled", "must", "may", "shall", "includes", "available", "eligible", "not allowed")
        ):
            lexical_bonus += 0.1
        if previous_line.endswith("?"):
            prev_terms = support_terms(previous_line)
            lexical_bonus += 0.15 * (len(query_terms & prev_terms) / max(1, len(query_terms))) if query_terms else 0.0
        if index <= 2:
            lexical_bonus += 0.05

        score = overlap + lexical_bonus
        explicit_overlap = overlap >= 0.34 or (
            query_terms and len(query_terms & line_terms) >= max(1, min(2, len(query_terms)))
        )
        scored_lines.append(
            {
                "text": stripped,
                "score": score,
                "mode": "explicit" if explicit_overlap else "interpretive",
            }
        )

        previous_line = stripped

    if not scored_lines:
        return {"text": "", "lines": [], "mode": "interpretive", "score": 0.0}

    scored_lines.sort(key=lambda item: float(item["score"]), reverse=True)
    best_score = float(scored_lines[0]["score"])
    selected_lines = [
        item
        for item in scored_lines
        if float(item["score"]) >= max(0.3, best_score - 0.12)
    ][:3]

    combined_text = "\n".join(f"- {item['text']}" for item in selected_lines)
    explicit_count = sum(1 for item in selected_lines if item["mode"] == "explicit")
    combined_mode = "explicit" if explicit_count >= max(1, len(selected_lines) // 2) else "interpretive"

    return {
        "text": combined_text,
        "lines": [str(item["text"]) for item in selected_lines],
        "mode": combined_mode,
        "score": round(best_score, 4),
    }


def select_simple_context_chunks(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_chunks: int = 2,
) -> list[RetrievedChunk]:
    """Use rank 1 as the anchor and add one supporting chunk only if it stays on the same topic."""
    if not retrieved_chunks:
        return []
    if len(retrieved_chunks) == 1 or max_chunks <= 1:
        return retrieved_chunks[:1]

    selected = [retrieved_chunks[0]]
    query_terms = support_terms(user_query)
    top_terms = support_terms(retrieved_chunks[0].get("chunk_text", ""))

    for candidate in retrieved_chunks[1: min(len(retrieved_chunks), max_chunks + 2)]:
        candidate_terms = support_terms(candidate.get("chunk_text", ""))
        if not candidate_terms:
            continue

        query_overlap = (
            len(query_terms & candidate_terms) / max(1, len(query_terms))
            if query_terms else 0.0
        )
        topic_overlap = (
            len(top_terms & candidate_terms) / max(1, len(top_terms))
            if top_terms else 0.0
        )
        combined_score = max(
            float(candidate.get("score", 0.0)),
            float(candidate.get("keyword_overlap", 0.0)),
            float(candidate.get("reranker_score", float("-inf"))) if candidate.get("reranker_score") is not None else 0.0,
        )

        if query_overlap >= 0.2 and topic_overlap >= 0.12 and combined_score >= 0.3:
            selected.append(candidate)
        if len(selected) >= max_chunks:
            break

    return selected


def needs_interpretive_rewrite(
    user_query: str,
    answer: str,
    primary_evidence_payload: dict[str, Any],
) -> bool:
    """Detect when an interpretive answer overcommits beyond the explicit evidence."""
    if not answer.strip():
        return False
    if primary_evidence_payload.get("mode") == "explicit":
        return False

    evidence_text = " ".join(primary_evidence_payload.get("lines", [])).lower()
    query_terms = [term for term in support_terms(user_query) if len(term) >= 4]
    if not query_terms:
        return False

    missing_terms = [term for term in query_terms if term not in evidence_text]
    if not missing_terms:
        return False

    lowered_answer = answer.strip().lower()
    if len(lowered_answer.split()) <= 8 and not lowered_answer.startswith(("yes", "no", "you can", "you cannot", "you can't")):
        return False

    definitive_patterns = (
        lowered_answer.startswith("yes")
        or lowered_answer.startswith("no")
        or lowered_answer.startswith("you can")
        or lowered_answer.startswith("you cannot")
        or lowered_answer.startswith("you can't")
        or lowered_answer.startswith("you may")
        or lowered_answer.startswith("you may not")
        or lowered_answer.startswith("you are allowed")
        or lowered_answer.startswith("you are not allowed")
        or " is allowed" in lowered_answer
        or " is not allowed" in lowered_answer
        or " can " in lowered_answer
        or " cannot " in lowered_answer
    )
    return definitive_patterns


def build_interpretive_rewrite_prompt(
    user_query: str,
    answer: str,
    primary_evidence_payload: dict[str, Any],
) -> str:
    """Rewrite an overcommitted interpretive answer into conditional grounded wording."""
    evidence_block = primary_evidence_payload.get("text", "").strip()
    return f"""You are revising a document-grounded answer.
Do not use outside knowledge.
Answer in English.

The current answer is too definite for the available evidence.
Rewrite it so that:
- it does not use an unqualified yes/no
- it does not claim permission or prohibition unless explicitly stated in the evidence
- if a qualification is needed, keep it brief and natural
- it uses cautious conditional wording such as "would depend on" or "the document only states"
- it stays concise and natural
- it avoids repetitive stock phrases such as "the exact wording is not explicitly stated" unless absolutely necessary
- output only the final answer text
- do not include any preface like "Here's a revised answer"
- do not include notes, explanations, bullet points, or commentary about your rewrite

User question:
{user_query.strip()}

Current answer:
{answer.strip()}

Primary evidence:
{evidence_block}
"""


def rewrite_interpretive_answer_with_ollama(
    user_query: str,
    answer: str,
    primary_evidence_payload: dict[str, Any],
    *,
    model_name: str,
) -> str:
    """Use the LLM to soften an overcommitted interpretation into a grounded conditional answer."""
    prompt = build_interpretive_rewrite_prompt(
        user_query,
        answer,
        primary_evidence_payload,
    )
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 160,
            "top_p": 0.9,
        },
    )
    rewritten = (response_json.get("response") or "").strip()
    return rewritten or answer


def build_query_rewrite_prompt(user_query: str) -> str:
    """Ask the LLM for lightweight semantic retrieval terms only."""
    return f"""You rewrite user questions for document retrieval.
Do not answer the question.
Do not explain.
Use general knowledge only to identify short, generic semantic retrieval terms.

Rules:
- Do not rewrite the full question.
- Return only 2 to 4 short semantic terms or phrases that help retrieval.
- Preserve the user's core topic and important concrete terms.
- You may add closely related topic terms if they improve retrieval.
- Keep the semantic terms in English.
- Do not add assumptions about gender, age, role, department, location, or audience unless explicitly stated.
- Do not invent facts or assume the answer.
- Return JSON only.

Output JSON schema:
{{
  "semantic_terms": ["term1", "term2", "term3"]
}}

User question:
{user_query.strip()}
"""


def build_query_plan_prompt(user_query: str) -> str:
    """Ask the LLM whether the question is simple or complex and optionally decompose it."""
    return f"""You are a query planner for document question answering.
Do not answer the question.
Classify the query as:
- "simple": one main information need that can likely be answered from one focused retrieval
- "complex": multiple distinct information needs, comparison, combination, or multi-step reasoning

If the query is complex, break it into 2 to 4 short sub-queries that preserve the user's meaning.
If the query is simple, return an empty sub_queries list.

Rules:
- Do not add assumptions not present in the question.
- Do not expand the scope beyond what the user asked.
- Keep sub-queries short and retrieval-friendly.
- Keep sub-queries in English.
- Return JSON only.

Output schema:
{{
  "complexity": "simple" | "complex",
  "reason": "short reason",
  "sub_queries": ["sub query 1", "sub query 2"]
}}

User question:
{user_query.strip()}
"""


def rewrite_query_with_ollama(
    user_query: str,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Use the LLM to produce lightweight semantic terms while keeping the raw query dominant."""
    prompt = build_query_rewrite_prompt(user_query)
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 120,
            "top_p": 0.9,
        },
    )
    raw_response = (response_json.get("response") or "").strip()
    parsed = _parse_query_rewrite_json(raw_response)
    semantic_terms = parsed.get("semantic_terms", [])
    if not isinstance(semantic_terms, list):
        semantic_terms = []
    semantic_terms = [
        " ".join(str(term).split()).strip()
        for term in semantic_terms
        if str(term).strip()
    ][:6]
    semantic_terms = _dedupe_terms_preserve_order(semantic_terms)
    return {
        "raw_response": raw_response,
        "semantic_terms": semantic_terms,
    }


def plan_query_with_ollama(
    user_query: str,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Use the LLM to decide whether a query is simple or complex."""
    prompt = build_query_plan_prompt(user_query)
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 180,
            "top_p": 0.9,
        },
    )
    raw_response = (response_json.get("response") or "").strip()
    parsed = _parse_query_rewrite_json(raw_response)

    complexity = str(parsed.get("complexity", "simple")).strip().lower()
    if complexity not in {"simple", "complex"}:
        complexity = "simple"

    sub_queries = parsed.get("sub_queries", [])
    if not isinstance(sub_queries, list):
        sub_queries = []
    sub_queries = [
        " ".join(str(query).split()).strip()
        for query in sub_queries
        if str(query).strip()
    ][:4]
    sub_queries = _dedupe_terms_preserve_order(sub_queries)

    reason = " ".join(str(parsed.get("reason", "")).split()).strip()
    if complexity == "simple":
        sub_queries = []

    return {
        "raw_response": raw_response,
        "complexity": complexity,
        "reason": reason,
        "sub_queries": sub_queries,
    }


def _parse_query_rewrite_json(raw_text: str) -> dict[str, Any]:
    """Parse a best-effort JSON object from an LLM rewrite response."""
    if not raw_text:
        return {}
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw_text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _dedupe_terms_preserve_order(terms: list[str]) -> list[str]:
    """Deduplicate semantic terms while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(term)
    return deduped


def build_grounded_interpretation_prompt(
    user_query: str,
    context: str,
    retrieval_query: str | None = None,
) -> str:
    """Create a conservative interpretation prompt anchored to a very small evidence set."""
    query_block = user_query
    if retrieval_query and retrieval_query.strip() and retrieval_query.strip() != user_query.strip():
        query_block = (
            f"Original user question: {user_query.strip()}\n"
            f"Retrieval interpretation: {retrieval_query.strip()}"
        )

    return f"""You are a document-grounded policy assistant.
Use only the provided context.
Answer in English.

If the document explicitly answers the question, answer directly.
If the wording differs but the document still answers the question in substance, give a concise grounded paraphrase.
Only add a short qualification when it is genuinely necessary.
Do not invent permissions, exceptions, or requirements that are not in the context.
If the context is insufficient, respond exactly with:
{FALLBACK_ANSWER}

Keep the answer concise and factual.
Stay on the same topic as the user's question.

User question:
{query_block}

Context:
{context}
"""


def build_subanswer_combine_prompt(
    user_query: str,
    sub_answers: list[dict[str, str]],
) -> str:
    """Combine sub-answers into one final grounded answer."""
    sub_blocks: list[str] = []
    for item in sub_answers:
        query = item.get("query", "").strip()
        answer = item.get("answer", "").strip()
        facts = item.get("facts", "").strip()
        if facts:
            sub_blocks.append(f"- Sub-query: {query}\n  Supported facts:\n{facts}")
        elif answer:
            sub_blocks.append(f"- Sub-query: {query}\n  Answer: {answer}")
    sub_answer_block = "\n".join(sub_blocks)
    return f"""You are combining document-grounded sub-answers.
Use only the sub-answers below.
Do not add outside knowledge.
Answer in English.
Write one concise final answer to the original user question.
Cover each sub-query that has supported information.
If one sub-answer lacks evidence, omit unsupported details rather than inventing them.
Do not say that the wording is "not explicitly stated" if the sub-answers already contain explicit policy details.
Prefer a compact factual summary over a cautious interpretation.
If supported facts are provided, prioritize those facts over any looser narrative wording.

Original user question:
{user_query.strip()}

Sub-answers:
{sub_answer_block}
"""


def _is_index_like_chunk(text: str) -> bool:
    """Detect table-of-contents or heading-index style chunks that are poor evidence for summaries."""
    lowered = text.lower()
    if "table of contents" in lowered:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    heading_like = 0
    short_lines = 0
    for line in lines[:24]:
        if len(line.split()) <= 8:
            short_lines += 1
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[a-z]", line.lower()):
            heading_like += 1

    line_count = len(lines[:24])
    if line_count >= 8 and heading_like / line_count >= 0.45 and short_lines / line_count >= 0.55:
        return True
    return False


def _prose_quality_score(text: str) -> float:
    """Estimate whether a chunk contains substantive policy prose rather than a sparse index."""
    words = text.split()
    if not words:
        return 0.0

    sentence_count = len(re.findall(r"[.!?]", text))
    longish_lines = sum(1 for line in text.splitlines() if len(line.split()) >= 10)
    base = min(0.25, len(words) / 500.0)
    base += min(0.18, sentence_count * 0.03)
    base += min(0.12, longish_lines * 0.02)
    if _is_index_like_chunk(text):
        base -= 0.35
    return base


def select_summary_context_chunks(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_chunks: int = 3,
) -> list[RetrievedChunk]:
    """Keep a slightly broader evidence set for factual topic summaries."""
    if not retrieved_chunks:
        return []

    query_terms = support_terms(user_query)
    scored: list[tuple[float, RetrievedChunk]] = []
    for index, chunk in enumerate(retrieved_chunks[:5]):
        chunk_text = chunk.get("chunk_text", "")
        if _is_index_like_chunk(chunk_text):
            continue
        chunk_terms = support_terms(chunk_text)
        overlap = len(query_terms & chunk_terms) / max(1, len(query_terms)) if query_terms else 0.0
        score = (
            overlap
            + 0.35 * float(chunk.get("keyword_overlap", 0.0))
            + min(0.2, float(chunk.get("score", 0.0)) * 0.2)
            + _prose_quality_score(chunk_text)
            + max(0.0, 0.06 - (index * 0.01))
        )
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored[: max(1, max_chunks)]]
    return selected or [chunk for chunk in retrieved_chunks[:max_chunks] if not _is_index_like_chunk(chunk.get("chunk_text", ""))] or retrieved_chunks[: max(1, max_chunks)]


def select_topic_evidence_lines(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_lines: int = 6,
) -> dict[str, Any]:
    """Select multiple explicit policy lines across the top chunks for summary questions."""
    query_terms = support_terms(user_query)
    candidates: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(retrieved_chunks[:4]):
        structured_lines = prepare_structured_lines(chunk.get("chunk_text", ""))
        if _is_index_like_chunk(chunk.get("chunk_text", "")):
            continue
        heading = ""
        for line_index, line in enumerate(structured_lines):
            stripped = line.strip()
            if not stripped or stripped.endswith("?"):
                continue
            lowered = stripped.lower()
            if len(stripped.split()) <= 8 and (stripped.endswith(":") or re.match(r"^\d+(?:\.\d+)*\.?\s+", stripped)):
                heading = stripped
                continue

            line_terms = support_terms(stripped)
            if not line_terms:
                continue

            overlap = len(query_terms & line_terms) / max(1, len(query_terms)) if query_terms else 0.0
            if heading:
                heading_terms = support_terms(heading)
                overlap = max(overlap, len(query_terms & heading_terms) / max(1, len(query_terms)) if query_terms else overlap)

            cue_bonus = 0.0
            if any(cue in lowered for cue in ("entitled", "required", "must", "shall", "will", "eligible", "notice", "leave", "resignation", "final wages")):
                cue_bonus += 0.14
            if re.search(r"\b\d+(?:\.\d+)?\b", lowered):
                cue_bonus += 0.08
            if "not" in lowered and "not be" in lowered:
                cue_bonus += 0.04

            score = overlap + cue_bonus + max(0.0, 0.05 - (chunk_index * 0.01) - (line_index * 0.004))
            if score < 0.16:
                continue

            candidates.append(
                {
                    "text": stripped,
                    "score": score,
                    "chunk": chunk,
                }
            )

    if not candidates:
        return {"text": "", "lines": [], "chunks": []}

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    selected_lines: list[str] = []
    selected_chunks: list[RetrievedChunk] = []
    seen_lines: set[str] = set()
    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        text = candidate["text"]
        normalized = text.lower()
        if normalized in seen_lines:
            continue
        seen_lines.add(normalized)
        selected_lines.append(text)
        chunk = candidate["chunk"]
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            selected_chunks.append(chunk)
        if len(selected_lines) >= max_lines:
            break

    return {
        "text": "\n".join(f"- {line}" for line in selected_lines),
        "lines": selected_lines,
        "chunks": selected_chunks,
    }


def build_factual_topic_summary_prompt(
    user_query: str,
    evidence_lines: str,
    context: str,
) -> str:
    """Prompt for strict factual summarization from explicit evidence."""
    return f"""You are a document-grounded assistant.
Use only the provided evidence and context.
Do not use outside knowledge.
Answer in English.

Summarize only rules or facts that are explicitly supported by the evidence.
Do not say the topic is "not explicitly stated" if the evidence already contains direct policy details.
Do not narrow the answer to only one subsection if multiple evidence lines are present.
Write a concise factual summary in 2 to 4 sentences.
If the evidence is truly insufficient, respond exactly with:
{FALLBACK_ANSWER}

User question:
{user_query.strip()}

Primary evidence:
{evidence_lines.strip()}

Supporting context:
{context}
"""


def build_factual_topic_extraction_prompt(
    user_query: str,
    evidence_lines: str,
    context: str,
) -> str:
    """Prompt for extracting explicit supported facts as bullets."""
    return f"""You are a document-grounded assistant.
Use only the provided evidence and context.
Do not use outside knowledge.
Answer in English.

Extract only explicit supported facts or rules relevant to the user question.
Return 3 to 6 short bullet points.
Each bullet must state a concrete rule, entitlement, requirement, timeline, amount, or restriction from the evidence.
Do not include introductory text.
Do not include commentary such as "not explicitly stated" if direct policy details already exist.
Do not infer beyond the text.
If the evidence is truly insufficient, respond exactly with:
{FALLBACK_ANSWER}

User question:
{user_query.strip()}

Primary evidence:
{evidence_lines.strip()}

Supporting context:
{context}
"""


def extract_topic_facts_with_ollama(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    max_context_chunks: int = 4,
    max_chars_per_chunk: int = 1100,
    temperature: float = 0.0,
    num_predict: int = 180,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Extract explicit bullet facts for broad factual topic summaries."""
    fallback_answer = FALLBACK_ANSWER
    context_chunks = select_summary_context_chunks(
        user_query,
        retrieved_chunks,
        max_chunks=min(4, max_context_chunks),
    )
    context = build_context_from_chunks_limited(
        context_chunks,
        max_context_chunks=min(4, max_context_chunks),
        max_chars_per_chunk=min(1100, max_chars_per_chunk),
    )
    citations = format_citations(context_chunks)
    if not context.strip():
        response_payload = attach_citations_to_response(fallback_answer, [])
        return {
            "answer": fallback_answer,
            "facts": "",
            "fact_lines": [],
            "sources": [],
            "citations": response_payload["citations"],
            "citations_text": response_payload["citations_text"],
            "context_used": context,
            "answer_strategy": "topic_fact_extraction_no_context",
            "primary_evidence": None,
            "raw_generated_answer": fallback_answer,
        }

    evidence_payload = select_topic_evidence_lines(
        user_query,
        context_chunks,
        max_lines=8,
    )
    prompt = build_factual_topic_extraction_prompt(
        user_query,
        evidence_payload.get("text", ""),
        context,
    )
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": min(num_predict, 180),
            "top_p": top_p,
        },
    )
    raw_facts = (response_json.get("response") or "").strip() or fallback_answer
    if raw_facts == fallback_answer:
        fact_lines: list[str] = []
    else:
        fact_lines = []
        for line in raw_facts.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[-*•]\s*", "", stripped)
            stripped = re.sub(r"^\d+\.\s*", "", stripped)
            if not stripped or stripped.lower() == fallback_answer.lower():
                continue
            fact_lines.append(stripped)
        deduped_lines: list[str] = []
        seen: set[str] = set()
        for line in fact_lines:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_lines.append(line)
        fact_lines = deduped_lines[:6]

    facts_text = "\n".join(f"- {line}" for line in fact_lines) if fact_lines else fallback_answer
    response_payload = attach_citations_to_response(facts_text, citations)
    return {
        "answer": facts_text,
        "facts": facts_text if facts_text != fallback_answer else "",
        "fact_lines": fact_lines,
        "sources": get_unique_sources(context_chunks) if citations else [],
        "citations": response_payload["citations"],
        "citations_text": response_payload["citations_text"],
        "context_used": context,
        "answer_strategy": "topic_fact_extraction",
        "primary_evidence": evidence_payload,
        "raw_generated_answer": raw_facts,
    }


def summarize_topic_with_ollama(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    max_context_chunks: int = 3,
    max_chars_per_chunk: int = 1000,
    temperature: float = 0.05,
    num_predict: int = 180,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Generate a factual topic summary from multiple explicit evidence lines."""
    fallback_answer = FALLBACK_ANSWER
    context_chunks = select_summary_context_chunks(
        user_query,
        retrieved_chunks,
        max_chunks=min(3, max_context_chunks),
    )
    context = build_context_from_chunks_limited(
        context_chunks,
        max_context_chunks=min(3, max_context_chunks),
        max_chars_per_chunk=min(1000, max_chars_per_chunk),
    )
    citations = format_citations(context_chunks)
    if not context.strip():
        response_payload = attach_citations_to_response(fallback_answer, [])
        return {
            "answer": fallback_answer,
            "sources": [],
            "citations": response_payload["citations"],
            "citations_text": response_payload["citations_text"],
            "context_used": context,
            "answer_strategy": "topic_summary_no_context",
            "primary_evidence": None,
            "raw_generated_answer": fallback_answer,
        }

    evidence_payload = select_topic_evidence_lines(user_query, context_chunks)
    prompt = build_factual_topic_summary_prompt(
        user_query,
        evidence_payload.get("text", ""),
        context,
    )
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": min(num_predict, 180),
            "top_p": top_p,
        },
    )
    answer = (response_json.get("response") or "").strip() or fallback_answer
    response_payload = attach_citations_to_response(answer, citations)
    return {
        "answer": response_payload["answer"],
        "sources": get_unique_sources(context_chunks) if citations else [],
        "citations": response_payload["citations"],
        "citations_text": response_payload["citations_text"],
        "context_used": context,
        "answer_strategy": "topic_summary_generation",
        "primary_evidence": evidence_payload,
        "raw_generated_answer": answer,
    }


def generate_answer_with_ollama(
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    model_name: str = DEFAULT_OLLAMA_MODEL,
    retrieval_query: str | None = None,
    max_context_chunks: int = 4,
    max_chars_per_chunk: int = 1200,
    temperature: float = 0.1,
    num_predict: int = 250,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Generate a grounded answer from retrieved chunks using a local Ollama model."""
    fallback_answer = FALLBACK_ANSWER
    answer_strategy = "top_rank_llm_generation"
    effective_max_context_chunks = min(max_context_chunks, 2)
    effective_max_chars_per_chunk = min(max_chars_per_chunk, 900)
    context_chunks = select_simple_context_chunks(
        user_query,
        retrieved_chunks,
        max_chunks=effective_max_context_chunks,
    )
    if len(context_chunks) > 1:
        answer_strategy = "top_rank_plus_support_generation"

    context = build_context_from_chunks_limited(
        context_chunks,
        max_context_chunks=effective_max_context_chunks,
        max_chars_per_chunk=effective_max_chars_per_chunk,
    )
    citations = format_citations(context_chunks)

    if not context.strip():
        response_payload = attach_citations_to_response(fallback_answer, [])
        return {
            "answer": fallback_answer,
            "sources": [],
            "citations": response_payload["citations"],
            "citations_text": response_payload["citations_text"],
            "context_used": context,
            "answer_strategy": "fallback_no_context",
        }

    primary_evidence_payload = select_primary_evidence(user_query, context_chunks[0] if context_chunks else None)
    prompt = build_basic_rag_prompt(
        user_query,
        context,
        retrieval_query=retrieval_query,
        primary_evidence=primary_evidence_payload.get("text"),
        evidence_mode=primary_evidence_payload.get("mode", "interpretive"),
    )
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": min(num_predict, 180),
            "top_p": top_p,
        },
    )
    raw_generated_answer = (response_json.get("response") or "").strip() or fallback_answer
    answer = raw_generated_answer
    if needs_interpretive_rewrite(user_query, answer, primary_evidence_payload):
        answer = rewrite_interpretive_answer_with_ollama(
            user_query,
            answer,
            primary_evidence_payload,
            model_name=model_name,
        )
    answer_quality = {
        "off_topic": False,
        "reason": (
            "Interpretive answer was rewritten into safer conditional wording."
            if answer != raw_generated_answer
            else "No post-generation rejection was needed."
        ),
        "unsupported_claims": [],
        "query_overlap": None,
        "top_chunk_overlap": None,
    }

    # Grounding rescue is temporarily disabled so the raw LLM answer is returned as-is.
    # should_rescue = answer == FALLBACK_ANSWER
    # if use_structured_controls and not should_rescue:
    #     answer_quality = assess_answer_quality(
    #         user_query,
    #         answer,
    #         context_chunks,
    #         query_intent=query_intent,
    #         fallback_answer=FALLBACK_ANSWER,
    #     )
    #     should_rescue = answer_quality["off_topic"]
    #
    # if should_rescue and use_structured_controls:
    #     rescued_answer = build_extractive_rescue_answer(
    #         user_query,
    #         context_chunks,
    #         query_intent=query_intent,
    #     )
    #     if rescued_answer:
    #         answer = rescued_answer
    #         answer_strategy = "extractive_rescue"
    #     else:
    #         citations = []

    response_payload = attach_citations_to_response(answer, citations)
    return {
        "answer": response_payload["answer"],
        "sources": get_unique_sources(context_chunks) if citations else [],
        "citations": response_payload["citations"],
        "citations_text": response_payload["citations_text"],
        "context_used": context,
        "answer_strategy": answer_strategy,
        "raw_generated_answer": raw_generated_answer,
        "answer_quality": answer_quality,
        "primary_evidence": primary_evidence_payload,
    }


def answer_complex_query_with_subqueries(
    user_query: str,
    *,
    sub_queries: list[str],
    index: Any,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    lexical_index: dict[str, Any] | None,
    llm_model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_chars_per_chunk: int,
    temperature: float,
    num_predict: int,
    top_p: float,
    reranker_model: Any,
    use_reranker: bool,
) -> dict[str, Any]:
    """Answer a complex query by answering smaller sub-queries first, then combining them."""
    sub_answers: list[dict[str, Any]] = []
    combined_citations: list[dict[str, Any]] = []
    combined_chunks: list[RetrievedChunk] = []
    retrieval_debugging: list[dict[str, Any]] = []

    for sub_query in sub_queries[:4]:
        sub_active_query = normalize_query(sub_query) or sub_query.strip()
        sub_chunks, sub_query_variants = retrieve_multi_query_chunks(
            original_query=sub_query,
            active_query=sub_active_query,
            canonical_query=None,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            lexical_index=lexical_index,
            top_k=top_k,
            reranker_model=reranker_model,
            use_reranker=use_reranker,
        )
        sub_evaluation = evaluate_retrieval_quality(sub_active_query, sub_chunks)
        retrieval_debugging.append(
            {
                "attempt": len(retrieval_debugging) + 1,
                "query": sub_query,
                "query_variants": sub_query_variants,
                "evaluation": sub_evaluation,
                "chunks": sub_chunks,
                "sub_query": True,
            }
        )

        if not sub_chunks:
            sub_answers.append(
                {
                    "query": sub_query,
                    "answer": FALLBACK_ANSWER,
                }
            )
            continue

        sub_answer_payload = generate_answer_with_ollama(
            sub_query,
            sub_chunks,
            model_name=llm_model_name,
            retrieval_query=sub_active_query,
            max_context_chunks=max_context_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
            temperature=temperature,
            num_predict=num_predict,
            top_p=top_p,
        )
        sub_answers.append(
            {
                "query": sub_query,
                "answer": sub_answer_payload["answer"],
            }
        )
        combined_citations.extend(sub_answer_payload.get("citations", []))
        combined_chunks.extend(sub_chunks[:1])

    combine_prompt = build_subanswer_combine_prompt(user_query, sub_answers)
    combine_response = _generate_with_ollama(
        model_name=llm_model_name,
        prompt=combine_prompt,
        options={
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": top_p,
        },
    )
    final_answer = (combine_response.get("response") or "").strip() or FALLBACK_ANSWER
    deduped_citations: list[dict[str, Any]] = []
    seen_citations: set[tuple[str, int, str]] = set()
    for citation in combined_citations:
        key = (
            str(citation.get("source", "")),
            int(citation.get("page_number", 0)),
            str(citation.get("chunk_id", "")),
        )
        if key in seen_citations:
            continue
        seen_citations.add(key)
        deduped_citations.append(citation)

    combined_context = build_context_from_chunks_limited(
        combined_chunks,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    response_payload = attach_citations_to_response(final_answer, deduped_citations)
    return {
        "answer": response_payload["answer"],
        "sources": get_unique_sources(combined_chunks) if deduped_citations else [],
        "citations": response_payload["citations"],
        "citations_text": response_payload["citations_text"],
        "context_used": combined_context,
        "answer_strategy": "complex_subquery_workflow",
        "raw_generated_answer": final_answer,
        "answer_quality": {
            "off_topic": False,
            "reason": "Answer composed from sub-query answers.",
            "unsupported_claims": [],
            "query_overlap": None,
            "top_chunk_overlap": None,
        },
        "primary_evidence": None,
        "sub_answers": sub_answers,
        "retrieval_debugging": retrieval_debugging,
    }


def evaluate_retrieval_quality(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
) -> dict[str, Any]:
    """Evaluate retrieval quality using chunk count, scores, and generic overlap signals."""
    if not retrieved_chunks:
        return {
            "is_relevant": False,
            "has_partial_evidence": False,
            "average_score": 0.0,
            "average_keyword_overlap": 0.0,
            "average_reranker_score": None,
            "reason": "No chunks were retrieved.",
            "chunk_count": 0,
        }

    chunk_count = len(retrieved_chunks)
    average_score = sum(float(chunk.get("score", 0.0)) for chunk in retrieved_chunks) / chunk_count
    keyword_overlaps = [
        float(chunk.get("keyword_overlap", calculate_keyword_overlap(query, chunk.get("chunk_text", ""))))
        for chunk in retrieved_chunks
    ]
    average_keyword_overlap = sum(keyword_overlaps) / chunk_count if keyword_overlaps else 0.0

    reranker_scores = [
        float(chunk["reranker_score"])
        for chunk in retrieved_chunks
        if chunk.get("reranker_score") is not None
    ]
    average_reranker_score = (
        sum(reranker_scores) / len(reranker_scores) if reranker_scores else None
    )

    meaningful_chunks = [
        chunk for chunk in retrieved_chunks if len((chunk.get("chunk_text") or "").split()) >= 25
    ]
    strong_faiss_hit = any(float(chunk.get("faiss_score", 0.0)) >= 0.38 for chunk in retrieved_chunks)
    strong_reranker_hit = any(
        chunk.get("reranker_score") is not None and float(chunk.get("reranker_score")) >= 0.0
        for chunk in retrieved_chunks
    )
    partial_overlap = average_keyword_overlap >= 0.03

    has_partial_evidence = bool(meaningful_chunks) and (partial_overlap or strong_faiss_hit or strong_reranker_hit)
    is_relevant = (
        len(meaningful_chunks) >= 2
        and (
            average_score >= 0.3
            or average_keyword_overlap >= 0.08
            or (average_reranker_score is not None and average_reranker_score >= 0.0)
        )
    )

    if is_relevant:
        reason = "Retrieved chunks contain usable evidence for grounded answering."
    elif has_partial_evidence:
        reason = "Some relevant evidence exists, but retrieval is weaker or more mixed than ideal."
    elif len(meaningful_chunks) < 1:
        reason = "Retrieved chunks were too sparse or too short to support an answer."
    else:
        reason = "Retrieved evidence was not strong enough after evaluation."

    return {
        "is_relevant": is_relevant,
        "has_partial_evidence": has_partial_evidence,
        "average_score": round(average_score, 4),
        "average_keyword_overlap": round(average_keyword_overlap, 4),
        "average_reranker_score": (
            round(average_reranker_score, 4) if average_reranker_score is not None else None
        ),
        "reason": reason,
        "chunk_count": chunk_count,
    }


def retrieve_multi_query_chunks(
    *,
    original_query: str,
    active_query: str,
    canonical_query: str | None,
    index: Any,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    lexical_index: dict[str, Any] | None,
    top_k: int,
    rewrite_query_text: str | None = None,
    reranker_model: Any = None,
    use_reranker: bool = False,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Retrieve chunks across several query variants and merge them into one ranking."""
    query_variants = build_query_variants(
        original_query,
        active_query,
        canonical_query=canonical_query,
        rewrite_query_text=rewrite_query_text,
    )
    normalized_original = normalize_query(original_query)
    normalized_active = normalize_query(active_query)
    variant_weights: dict[str, float] = {}
    for variant in query_variants:
        lowered = variant.lower()
        if lowered == original_query.lower().strip():
            variant_weights[variant] = 1.0
        elif normalized_original and lowered == normalized_original.lower():
            variant_weights[variant] = 0.9
        elif normalized_active and lowered == normalized_active.lower():
            variant_weights[variant] = 0.85
        elif canonical_query and lowered == canonical_query.lower().strip():
            variant_weights[variant] = 0.55
        elif rewrite_query_text and lowered == rewrite_query_text.lower().strip():
            variant_weights[variant] = 0.65
        else:
            variant_weights[variant] = 0.75

    merged_chunks: dict[tuple[str, int, str], RetrievedChunk] = {}
    for variant in query_variants:
        variant_weight = variant_weights.get(variant, 0.75)
        retrieved = search_similar_chunks(
            variant,
            index,
            metadata,
            embedding_model,
            lexical_index=lexical_index,
            top_k=top_k,
        )
        for chunk in retrieved:
            key = (chunk["source"], chunk["page_number"], chunk["chunk_id"])
            existing = merged_chunks.get(key)
            candidate = dict(chunk)
            candidate["variant_weight"] = variant_weight
            candidate["score"] = float(candidate.get("score", 0.0)) * variant_weight
            candidate["matched_queries"] = sorted(
                set((existing or {}).get("matched_queries", [])) | {variant}
            )
            if existing is None or _chunk_sort_tuple(candidate) > _chunk_sort_tuple(existing):
                merged_chunks[key] = {
                    **candidate,
                    "score": max(float(candidate.get("score", 0.0)), float((existing or {}).get("score", 0.0))),
                    "semantic_score": max(
                        float(candidate.get("semantic_score", 0.0)),
                        float((existing or {}).get("semantic_score", 0.0)),
                    ),
                    "faiss_score": max(
                        float(candidate.get("faiss_score", 0.0)),
                        float((existing or {}).get("faiss_score", 0.0)),
                    ),
                    "lexical_score": max(
                        float(candidate.get("lexical_score", 0.0)),
                        float((existing or {}).get("lexical_score", 0.0)),
                    ),
                    "keyword_overlap": max(
                        float(candidate.get("keyword_overlap", 0.0)),
                        float((existing or {}).get("keyword_overlap", 0.0)),
                    ),
                    "variant_weight": max(
                        float(candidate.get("variant_weight", 0.0)),
                        float((existing or {}).get("variant_weight", 0.0)),
                    ),
                }
            else:
                existing["matched_queries"] = candidate["matched_queries"]

    fused_chunks = list(merged_chunks.values())
    fused_chunks.sort(key=_chunk_sort_tuple, reverse=True)
    fused_chunks = _assign_chunk_ranks(fused_chunks[: max(top_k * 2, top_k)])
    if use_reranker:
        fused_chunks = rerank_chunks(
            active_query,
            fused_chunks,
            reranker_model,
            top_n=top_k,
        )
    else:
        fused_chunks = _assign_chunk_ranks(fused_chunks[:top_k])

    return fused_chunks, query_variants


def run_agentic_rag_pipeline(
    user_query: str,
    index: Any,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    lexical_index: dict[str, Any] | None = None,
    llm_model_name: str = DEFAULT_OLLAMA_MODEL,
    top_k: int = 10,
    max_retries: int = 1,
    max_context_chunks: int = 4,
    max_chars_per_chunk: int = 1200,
    answer_temperature: float = 0.1,
    answer_num_predict: int = 250,
    answer_top_p: float = 0.9,
    use_reranker: bool = False,
    reranker_model: Any = None,
) -> dict[str, Any]:
    """Run a retrieval-first agentic RAG pipeline with optional reranking and bounded retry."""
    pipeline_started = time.perf_counter()
    original_query = user_query.strip()
    retrieval_time_seconds = 0.0
    rewrite_time_seconds = 0.0
    answer_generation_time_seconds = 0.0
    verification_time_seconds = 0.0
    query_intent = analyze_query_intent(original_query)
    normalized_query = query_intent.get("normalized_query", "") or normalize_query(original_query)
    query_plan = {
        "raw_response": "",
        "complexity": "simple",
        "reason": "",
        "sub_queries": [],
    }
    llm_query_rewrite = {
        "raw_response": "",
        "semantic_terms": [],
    }
    planner_started = time.perf_counter()
    query_plan = plan_query_with_ollama(
        original_query,
        model_name=llm_model_name,
    )
    rewrite_time_seconds += time.perf_counter() - planner_started
    rewrite_started = time.perf_counter()
    llm_query_rewrite = rewrite_query_with_ollama(
        original_query,
        model_name=llm_model_name,
    )
    rewrite_time_seconds += time.perf_counter() - rewrite_started
    semantic_term_suffix = " ".join(llm_query_rewrite.get("semantic_terms", []))
    canonical_query = semantic_term_suffix
    current_query = normalized_query or original_query
    retrieval_attempts = 0
    final_chunks: list[RetrievedChunk] = []
    retrieval_debugging: list[dict[str, Any]] = []
    evaluation = {
        "is_relevant": False,
        "has_partial_evidence": False,
        "average_score": 0.0,
        "average_keyword_overlap": 0.0,
        "average_reranker_score": None,
        "reason": "Retrieval not attempted.",
        "chunk_count": 0,
    }
    rewrite_used = False
    query_variants_used: list[str] = []
    answer_verification = {
        "verified": True,
        "repair_attempted": False,
        "unsupported_claims": [],
        "reason": "Verification not needed.",
    }

    for attempt in range(max_retries + 1):
        retrieval_attempts += 1
        retrieval_started = time.perf_counter()
        rewrite_candidate: str | None = None
        if attempt > 0 and rewrite_used:
            rewrite_candidate = current_query

        final_chunks, query_variants = retrieve_multi_query_chunks(
            original_query=original_query,
            active_query=current_query,
            canonical_query=canonical_query,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            lexical_index=lexical_index,
            rewrite_query_text=rewrite_candidate,
            top_k=top_k,
            reranker_model=reranker_model,
            use_reranker=use_reranker,
        )
        query_variants_used = query_variants
        retrieval_time_seconds += time.perf_counter() - retrieval_started

        evaluation = evaluate_retrieval_quality(current_query, final_chunks)
        retrieval_debugging.append(
            {
                "attempt": attempt + 1,
                "query": current_query,
                "query_variants": query_variants,
                "evaluation": evaluation,
                "chunks": final_chunks,
            }
        )

        if evaluation["is_relevant"] or (attempt == max_retries and evaluation["has_partial_evidence"]):
            break

        if attempt < max_retries and should_rewrite_query(original_query, evaluation):
            retry_parts = [
                normalized_query,
                semantic_term_suffix,
            ]
            current_query = " ".join(part for part in retry_parts if part).strip() or original_query
            rewrite_used = True
        elif attempt < max_retries:
            retry_parts = [
                current_query,
                normalized_query,
                semantic_term_suffix,
            ]
            current_query = " ".join(part for part in retry_parts if part).strip()
            rewrite_used = True

    should_fallback_immediately = not evaluation["is_relevant"] and not evaluation["has_partial_evidence"]
    if not rewrite_used and should_rewrite_query(original_query, evaluation):
        generation_query = current_query
    else:
        generation_query = current_query if rewrite_used and current_query.strip() else original_query

    if should_fallback_immediately:
        answer_payload = {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "citations": [],
            "citations_text": "",
            "context_used": build_context_from_chunks_limited(
                final_chunks,
                max_context_chunks=max_context_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            ),
            "answer_strategy": "fallback_no_evidence",
        }
        final_answer = FALLBACK_ANSWER
    else:
        answer_started = time.perf_counter()
        if query_plan.get("complexity") == "complex" and query_plan.get("sub_queries"):
            answer_payload = answer_complex_query_with_subqueries(
                original_query,
                sub_queries=query_plan.get("sub_queries", []),
                index=index,
                metadata=metadata,
                embedding_model=embedding_model,
                lexical_index=lexical_index,
                llm_model_name=llm_model_name,
                top_k=top_k,
                max_context_chunks=max_context_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
                temperature=answer_temperature,
                num_predict=answer_num_predict,
                top_p=answer_top_p,
                reranker_model=reranker_model,
                use_reranker=use_reranker,
            )
            retrieval_debugging.extend(answer_payload.get("retrieval_debugging", []))
        else:
            answer_payload = generate_answer_with_ollama(
                original_query,
                final_chunks,
                model_name=llm_model_name,
                retrieval_query=canonical_query or generation_query,
                max_context_chunks=max_context_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
                temperature=answer_temperature,
                num_predict=answer_num_predict,
                top_p=answer_top_p,
            )
        answer_generation_time_seconds = time.perf_counter() - answer_started
        final_answer = answer_payload["answer"]
        # Final grounding verification is temporarily disabled so the LLM answer
        # is returned directly without post-generation repair or fallback.
        # verification_started = time.perf_counter()
        # answer_verification = verify_answer_grounding(
        #     user_query=original_query,
        #     answer=final_answer,
        #     retrieved_chunks=final_chunks,
        #     model_name=llm_model_name,
        #     max_context_chunks=max_context_chunks,
        #     max_chars_per_chunk=max_chars_per_chunk,
        #     original_user_query=original_query,
        # )
        # verification_time_seconds = time.perf_counter() - verification_started
        # if answer_verification["repaired_answer"] != final_answer:
        #     final_answer = answer_verification["repaired_answer"]
        #     if final_answer == FALLBACK_ANSWER:
        #         answer_payload = {
        #             "answer": FALLBACK_ANSWER,
        #             "sources": [],
        #             "citations": [],
        #             "citations_text": "",
        #             "context_used": answer_payload["context_used"],
        #             "answer_strategy": answer_payload.get("answer_strategy", "verification_fallback"),
        #         }
        #     else:
        #         repaired_response = attach_citations_to_response(final_answer, format_citations(final_chunks))
        #         answer_payload = {
        #             "answer": repaired_response["answer"],
        #             "sources": get_unique_sources(final_chunks),
        #             "citations": repaired_response["citations"],
        #             "citations_text": repaired_response["citations_text"],
        #             "context_used": answer_payload["context_used"],
        #             "answer_strategy": answer_payload.get("answer_strategy", "verified_rewrite"),
        #         }

    fallback_triggered = final_answer == FALLBACK_ANSWER
    if fallback_triggered and evaluation["is_relevant"]:
        evaluation = {
            **evaluation,
            "reason": "Relevant chunks were retrieved, but the final answer still fell back because the evidence was not explicit enough.",
        }

    total_pipeline_time_seconds = time.perf_counter() - pipeline_started
    return {
        "original_query": original_query,
        "rewritten_query": current_query,
        "canonical_query": canonical_query,
        "llm_query_rewrite": llm_query_rewrite,
        "query_plan": query_plan,
        "query_intent": query_intent,
        "generation_query": generation_query,
        "query_variants_used": query_variants_used,
        "rewrite_used": rewrite_used,
        "retrieval_attempts": retrieval_attempts,
        "retrieval_evaluation": evaluation,
        "answer_verification": answer_verification,
        "retrieved_chunks": final_chunks,
        "final_answer": final_answer,
        "fallback_triggered": fallback_triggered,
        "sources": answer_payload["sources"],
        "citations": answer_payload["citations"],
        "citations_text": answer_payload["citations_text"],
        "context_used": answer_payload["context_used"],
        "answer_strategy": answer_payload.get("answer_strategy", "unknown"),
        "raw_generated_answer": answer_payload.get("raw_generated_answer"),
        "answer_quality": answer_payload.get("answer_quality"),
        "primary_evidence": answer_payload.get("primary_evidence"),
        "sub_answers": answer_payload.get("sub_answers", []),
        "retrieval_debugging": retrieval_debugging,
        "performance": {
            "retrieval_time_seconds": round(retrieval_time_seconds, 2),
            "query_rewrite_time_seconds": round(rewrite_time_seconds, 2),
            "answer_generation_time_seconds": round(answer_generation_time_seconds, 2),
            "answer_verification_time_seconds": round(verification_time_seconds, 2),
            "total_pipeline_time_seconds": round(total_pipeline_time_seconds, 2),
        },
    }


def get_unique_sources(retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Remove duplicate source/page/chunk references while preserving order."""
    unique_sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    for chunk in retrieved_chunks:
        key = (chunk["source"], chunk["page_number"], chunk["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(
            {
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    return unique_sources


def check_ollama_connection(model_name: str = DEFAULT_OLLAMA_MODEL) -> tuple[bool, str]:
    """Check whether the configured LLM provider is reachable and whether the requested model exists."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            return False, "Groq is configured as the LLM provider, but GROQ_API_KEY/GROQ_API is missing."
        try:
            response = requests.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, 10),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Could not connect to Groq at {GROQ_BASE_URL}: {exc}"

        models = response.json().get("data", [])
        available_model_names = {model.get("id", "") for model in models}
        if model_name not in available_model_names:
            return False, f"Groq is reachable, but model '{model_name}' is not available for this account."
        return True, f"Groq is reachable and model '{model_name}' is available."

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, 10),
        )
        response.raise_for_status()
    except requests.RequestException:
        return False, f"Could not connect to Ollama at {OLLAMA_BASE_URL}. Start Ollama and try again."

    models = response.json().get("models", [])
    available_model_names = {model.get("name", "") for model in models}
    if model_name not in available_model_names:
        return False, f"Ollama is running, but model '{model_name}' is not available. Pull it first."

    return True, f"Ollama is running and model '{model_name}' is available."


def verify_answer_grounding(
    *,
    user_query: str,
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    model_name: str,
    max_context_chunks: int,
    max_chars_per_chunk: int,
    original_user_query: str | None = None,
) -> dict[str, Any]:
    """Check answer claims against retrieved evidence and repair or reject if unsupported."""
    if answer == FALLBACK_ANSWER:
        return {
            "verified": True,
            "repair_attempted": False,
            "unsupported_claims": [],
            "reason": "Fallback answer does not require verification.",
            "repaired_answer": answer,
        }

    claims = extract_answer_claims(answer)
    unsupported_claims = [
        claim for claim in claims
        if not claim_has_support(claim, retrieved_chunks)
    ]
    if not unsupported_claims:
        return {
            "verified": True,
            "repair_attempted": False,
            "unsupported_claims": [],
            "reason": "All answer claims matched retrieved evidence.",
            "repaired_answer": answer,
        }

    if not ENABLE_LLM_ANSWER_REPAIR:
        if not STRICT_GROUNDING_FALLBACK:
            return {
                "verified": False,
                "repair_attempted": False,
                "unsupported_claims": unsupported_claims,
                "remaining_unsupported_claims": unsupported_claims,
                "reason": "Potentially unsupported claims were detected, but strict fallback is disabled.",
                "repaired_answer": answer,
            }
        return {
            "verified": False,
            "repair_attempted": False,
            "unsupported_claims": unsupported_claims,
            "remaining_unsupported_claims": unsupported_claims,
            "reason": "Unsupported claims were detected and answer repair is disabled.",
            "repaired_answer": FALLBACK_ANSWER,
        }

    repaired_answer = repair_answer_with_ollama(
        user_query=user_query,
        answer=answer,
        unsupported_claims=unsupported_claims,
        retrieved_chunks=retrieved_chunks,
        model_name=model_name,
        original_user_query=original_user_query,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    repaired_claims = extract_answer_claims(repaired_answer)
    remaining_unsupported = [
        claim for claim in repaired_claims
        if claim != FALLBACK_ANSWER and not claim_has_support(claim, retrieved_chunks)
    ]
    verified = not remaining_unsupported and repaired_answer != FALLBACK_ANSWER
    if not verified and len(remaining_unsupported) >= max(1, len(repaired_claims) // 2):
        repaired_answer = FALLBACK_ANSWER

    return {
        "verified": repaired_answer != FALLBACK_ANSWER and not remaining_unsupported,
        "repair_attempted": True,
        "unsupported_claims": unsupported_claims,
        "remaining_unsupported_claims": remaining_unsupported,
        "reason": (
            "Unsupported claims were repaired."
            if repaired_answer != FALLBACK_ANSWER and not remaining_unsupported
            else "Unsupported claims remained after repair; returning fallback."
        ),
        "repaired_answer": repaired_answer,
    }


def repair_answer_with_ollama(
    *,
    user_query: str,
    answer: str,
    unsupported_claims: list[str],
    retrieved_chunks: list[RetrievedChunk],
    model_name: str,
    original_user_query: str | None,
    max_context_chunks: int,
    max_chars_per_chunk: int,
) -> str:
    """Ask the LLM to rewrite an answer using only supported evidence."""
    context = build_context_from_chunks_limited(
        retrieved_chunks,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    unsupported_block = "\n".join(f"- {claim}" for claim in unsupported_claims)
    prompt = f"""You are verifying a document-grounded answer.
Rewrite the answer so every statement is supported by the provided context.
Remove unsupported claims.
Do not add outside knowledge.
If the context is insufficient, respond exactly with:
{FALLBACK_ANSWER}

User question:
{original_user_query or user_query}

Draft answer:
{answer}

Unsupported claims to remove or fix:
{unsupported_block}

Context:
{context}
"""
    response_json = _generate_with_ollama(
        model_name=model_name,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 220,
            "top_p": 0.9,
        },
    )
    rewritten_answer = (response_json.get("response") or "").strip()
    return rewritten_answer or FALLBACK_ANSWER


def _generate_with_ollama(
    *,
    model_name: str,
    prompt: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Call the configured LLM provider with consistent timeout and error handling."""
    if LLM_PROVIDER == "groq":
        return _generate_with_groq(
            model_name=model_name,
            prompt=prompt,
            options=options,
        )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, OLLAMA_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ReadTimeout as exc:
        raise OllamaServiceError(
            f"Ollama timed out while running model '{model_name}'. "
            "Use a smaller model such as 'llama3.2:3b', reduce output tokens, or increase the timeout.",
            status_code=504,
        ) from exc
    except requests.RequestException as exc:
        raise OllamaServiceError(
            f"Ollama request failed for model '{model_name}' at {OLLAMA_BASE_URL}: {exc}",
            status_code=502,
        ) from exc


def _generate_with_groq(
    *,
    model_name: str,
    prompt: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    if not GROQ_API_KEY:
        raise OllamaServiceError(
            "Groq is configured as the LLM provider, but GROQ_API_KEY/GROQ_API is missing.",
            status_code=500,
        )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": options.get("temperature", 0.1),
        "max_tokens": options.get("num_predict", 256),
        "top_p": options.get("top_p", 0.9),
        "stream": False,
    }
    try:
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, OLLAMA_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        response_json = response.json()
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return {
            "response": content,
            "provider_response": response_json,
        }
    except requests.exceptions.ReadTimeout as exc:
        raise OllamaServiceError(
            f"Groq timed out while running model '{model_name}'. Reduce prompt size or output tokens and try again.",
            status_code=504,
        ) from exc
    except requests.RequestException as exc:
        raise OllamaServiceError(
            f"Groq request failed for model '{model_name}' at {GROQ_BASE_URL}: {exc}",
            status_code=502,
        ) from exc


def _chunk_sort_tuple(chunk: RetrievedChunk) -> tuple[float, float, float, int]:
    """Sort chunks by combined score and supporting signals."""
    return (
        float(chunk.get("score", 0.0)),
        float(chunk.get("reranker_score", float("-inf"))),
        float(chunk.get("keyword_overlap", 0.0)),
        len(chunk.get("matched_queries", [])),
    )


def _assign_chunk_ranks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Assign 1-based ranks to merged retrieval chunks."""
    ranked_chunks: list[RetrievedChunk] = []
    for rank, chunk in enumerate(chunks, start=1):
        ranked_chunk = dict(chunk)
        ranked_chunk["rank"] = rank
        ranked_chunks.append(ranked_chunk)
    return ranked_chunks
