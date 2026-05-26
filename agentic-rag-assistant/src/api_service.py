"""Reusable backend service helpers for Streamlit and FastAPI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from src.agentic_rag import (
    DEFAULT_OLLAMA_MODEL,
    FALLBACK_ANSWER,
    _generate_with_ollama,
    build_subanswer_combine_prompt,
    evaluate_retrieval_quality,
    extract_topic_facts_with_ollama,
    generate_answer_with_ollama,
    get_unique_sources,
    plan_query_with_ollama,
    summarize_topic_with_ollama,
)
from src.citations import attach_citations_to_response
from src.chunking import chunk_pages, summarize_chunks
from src.document_loader import extract_text_from_pdf_file, load_pdfs_from_data_folder, summarize_extraction
from src.preprocessing import preprocess_pages, summarize_preprocessing
from src.llamaindex_backend import (
    build_llamaindex_index,
    llamaindex_is_available,
    retrieve_llamaindex_chunks,
)
from src.query_understanding import extract_query_topics, split_coordinated_question
from src.vector_store import (
    calculate_keyword_overlap,
    load_reranker_model,
    rerank_chunks,
    summarize_vector_store,
)
RAG_BACKEND = "llamaindex"


def _should_run_complex_planner(question: str) -> bool:
    """Use the planner only when the query shape is broad enough to justify the extra LLM call."""
    lowered = " ".join(question.lower().split())
    if not lowered:
        return False

    complex_markers = (
        " and ",
        " or ",
        "compare",
        "difference",
        "summarize",
        "summary",
        "policy for",
        "requirements for",
        "steps for",
        "process for",
        "what are",
        "explain",
    )
    token_count = len(lowered.split())
    return token_count >= 9 or any(marker in lowered for marker in complex_markers)


def _simple_path_limits(max_context_chunks: int, max_chars_per_chunk: int, answer_num_predict: int) -> tuple[int, int, int]:
    """Keep the simple path tighter so Ollama spends less time on long prompts."""
    return (
        min(max_context_chunks, 2),
        min(max_chars_per_chunk, 900),
        min(answer_num_predict, 180),
    )


def _complex_subanswer_limits(
    max_context_chunks: int,
    max_chars_per_chunk: int,
    answer_num_predict: int,
) -> tuple[int, int, int, int]:
    """Use smaller budgets for per-subquery answers and the final combine call."""
    return (
        min(max_context_chunks, 2),
        min(max_chars_per_chunk, 800),
        min(answer_num_predict, 160),
        min(answer_num_predict, 180),
    )


def _prepare_llamaindex_retrieved_chunks(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    reranker_model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Add overlap signals and always rerank the top LlamaIndex retrieval results."""
    if not retrieved_chunks:
        return []

    prepared: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        enriched = dict(chunk)
        enriched["keyword_overlap"] = float(
            chunk.get("keyword_overlap")
            or calculate_keyword_overlap(query, chunk.get("chunk_text", ""))
        )
        prepared.append(enriched)

    reranker_model = load_reranker_model(reranker_model_name)
    return rerank_chunks(
        query,
        prepared,
        reranker_model,
        top_n=top_k,
    )


def _build_forced_summary_plan(question: str) -> dict[str, Any] | None:
    """Force obvious multi-topic summaries into the complex path when the planner under-classifies them."""
    lowered = " ".join(question.lower().split())
    if not lowered:
        return None

    summary_markers = ("summarize", "summary", "overview")
    if not any(marker in lowered for marker in summary_markers):
        return None
    if " and " not in lowered and " or " not in lowered:
        return None

    topics = extract_query_topics(question)
    if len(topics) < 2:
        return None

    suffix = "rules"
    if "policy" in lowered or "policies" in lowered:
        suffix = "policy"
    elif "procedure" in lowered or "procedures" in lowered:
        suffix = "procedures"
    elif "requirement" in lowered or "requirements" in lowered:
        suffix = "requirements"

    sub_queries: list[str] = []
    for topic_terms in topics[:4]:
        topic_phrase = " ".join(sorted(topic_terms))
        sub_queries.append(f"{topic_phrase} {suffix}".strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for sub_query in sub_queries:
        lowered_sub = sub_query.lower()
        if lowered_sub in seen:
            continue
        seen.add(lowered_sub)
        deduped.append(sub_query)

    if len(deduped) < 2:
        return None

    return {
        "raw_response": "",
        "complexity": "complex",
        "reason": "Forced complex path for a coordinated summary query with multiple topics.",
        "sub_queries": deduped,
        "route_type": "summary_topics",
    }


def _build_forced_interpretation_plan(question: str) -> dict[str, Any] | None:
    """Force coordinated interpretation questions into item-specific subqueries."""
    lowered = " ".join(question.lower().split())
    if not lowered:
        return None

    interpretation_markers = (
        "can ",
        "can i ",
        "am i allowed",
        "is it okay",
        "is it allowed",
        "may i ",
    )
    if not any(lowered.startswith(marker) for marker in interpretation_markers):
        return None
    if " and " not in lowered and " or " not in lowered:
        return None

    sub_queries = split_coordinated_question(question)
    if len(sub_queries) < 2:
        return None

    return {
        "raw_response": "",
        "complexity": "complex",
        "reason": "Forced complex path for a coordinated interpretation query with multiple items.",
        "sub_queries": sub_queries,
        "route_type": "coordinated_interpretation",
    }

@dataclass
class PipelineArtifacts:
    """In-memory state for a processed document collection."""

    pages: list[dict[str, Any]]
    processed_pages: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    extraction_summary: dict[str, Any]
    preprocessing_summary: dict[str, Any]
    chunk_summary: dict[str, Any]
    vector_summary: dict[str, Any]
    vector_index: Any
    vector_metadata: list[dict[str, Any]]
    lexical_index: dict[str, Any]
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    source_names: list[str]
    rag_backend: str
    llamaindex_summary: dict[str, Any]
    llamaindex_index: Any
    llamaindex_documents: list[Any]


def process_uploaded_documents(
    uploaded_files: list[tuple[str, bytes]],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
) -> PipelineArtifacts:
    """Extract, preprocess, chunk, and index uploaded PDF bytes."""
    pages: list[dict[str, Any]] = []
    source_names: list[str] = []

    for filename, file_bytes in uploaded_files:
        source_names.append(filename)
        file_obj = BytesIO(file_bytes)
        file_obj.name = filename
        pages.extend(extract_text_from_pdf_file(file_obj, source_type="uploaded"))

    return _build_pipeline_artifacts(
        pages,
        source_names=source_names,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )


def process_local_documents(
    *,
    data_dir: str = "data",
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
) -> PipelineArtifacts:
    """Extract, preprocess, chunk, and index all PDFs in the local data directory."""
    data_path = Path(data_dir)
    source_names = sorted(path.name for path in data_path.glob("*.pdf"))
    pages = load_pdfs_from_data_folder(data_dir=data_dir)
    return _build_pipeline_artifacts(
        pages,
        source_names=source_names,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )


def run_semantic_retrieval(
    query: str,
    artifacts: PipelineArtifacts,
    *,
    top_k: int = 10,
    use_reranker: bool = True,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict[str, Any]]:
    """Run retrieval on an already-processed document state with the active LlamaIndex path."""
    if artifacts.llamaindex_index is None:
        return []

    retrieved_chunks = retrieve_llamaindex_chunks(
        query,
        index=artifacts.llamaindex_index,
        chunks=artifacts.chunks,
        top_k=top_k,
    )
    return _prepare_llamaindex_retrieved_chunks(
        query,
        retrieved_chunks,
        reranker_model_name=reranker_model_name,
        top_k=top_k,
    )


def run_agentic_answer(
    question: str,
    artifacts: PipelineArtifacts,
    *,
    llm_model_name: str = DEFAULT_OLLAMA_MODEL,
    top_k: int = 10,
    max_context_chunks: int = 4,
    max_chars_per_chunk: int = 1200,
    answer_temperature: float = 0.1,
    answer_num_predict: int = 250,
    use_reranker: bool = True,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, Any]:
    """Run the agentic RAG pipeline on an already-processed document state."""
    if artifacts.llamaindex_index is None:
        raise RuntimeError("LlamaIndex index is not available. Reprocess documents before asking questions.")

    return _run_llamaindex_agentic_answer(
        question,
        artifacts,
        llm_model_name=llm_model_name,
        top_k=top_k,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
        answer_temperature=answer_temperature,
        answer_num_predict=answer_num_predict,
        reranker_model_name=reranker_model_name,
    )


def _run_llamaindex_agentic_answer(
    question: str,
    artifacts: PipelineArtifacts,
    *,
    llm_model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_chars_per_chunk: int,
    answer_temperature: float,
    answer_num_predict: int,
    reranker_model_name: str,
) -> dict[str, Any]:
    """Run a LlamaIndex-backed retrieval path while preserving the current response shape."""
    pipeline_started = time.perf_counter()
    planner_time_seconds = 0.0
    if _should_run_complex_planner(question):
        planner_started = time.perf_counter()
        query_plan = plan_query_with_ollama(
            question,
            model_name=llm_model_name,
        )
        planner_time_seconds = time.perf_counter() - planner_started
    else:
        query_plan = {
            "raw_response": "",
            "complexity": "simple",
            "reason": "Planner skipped for a likely simple query.",
            "sub_queries": [],
        }

    forced_plan = _build_forced_summary_plan(question)
    if forced_plan is not None and query_plan.get("complexity") != "complex":
        query_plan = forced_plan
    forced_interpretation_plan = _build_forced_interpretation_plan(question)
    if forced_interpretation_plan is not None and query_plan.get("complexity") != "complex":
        query_plan = forced_interpretation_plan

    if query_plan.get("complexity") == "complex" and query_plan.get("sub_queries"):
        return _run_llamaindex_complex_answer(
            question,
            artifacts,
            llm_model_name=llm_model_name,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
            answer_temperature=answer_temperature,
            answer_num_predict=answer_num_predict,
            reranker_model_name=reranker_model_name,
            query_plan=query_plan,
            planner_time_seconds=planner_time_seconds,
            pipeline_started=pipeline_started,
        )

    simple_max_context_chunks, simple_max_chars_per_chunk, simple_num_predict = _simple_path_limits(
        max_context_chunks,
        max_chars_per_chunk,
        answer_num_predict,
    )
    retrieval_started = time.perf_counter()
    retrieved_chunks = retrieve_llamaindex_chunks(
        question,
        index=artifacts.llamaindex_index,
        chunks=artifacts.chunks,
        top_k=top_k,
    )
    retrieved_chunks = _prepare_llamaindex_retrieved_chunks(
        question,
        retrieved_chunks,
        reranker_model_name=reranker_model_name,
        top_k=top_k,
    )
    retrieval_time_seconds = time.perf_counter() - retrieval_started
    evaluation = evaluate_retrieval_quality(question, retrieved_chunks)

    if not retrieved_chunks:
        return {
            "original_query": question,
            "rewritten_query": question,
            "canonical_query": "",
            "llm_query_rewrite": {"raw_response": "", "semantic_terms": []},
            "query_plan": query_plan,
            "query_intent": {"normalized_query": question, "tokens": question.split()},
            "generation_query": question,
            "query_variants_used": [question],
            "rewrite_used": False,
            "retrieval_attempts": 1,
            "retrieval_evaluation": evaluation,
            "answer_verification": {
                "verified": True,
                "repair_attempted": False,
                "unsupported_claims": [],
                "reason": "Verification not needed.",
            },
            "retrieved_chunks": [],
            "final_answer": "I could not find this information in the uploaded documents.",
            "fallback_triggered": True,
            "sources": [],
            "citations": [],
            "citations_text": "",
            "context_used": "",
            "answer_strategy": "llamaindex_simple_workflow",
            "raw_generated_answer": "I could not find this information in the uploaded documents.",
            "answer_quality": {
                "off_topic": False,
                "reason": "No evidence was retrieved from LlamaIndex.",
                "unsupported_claims": [],
                "query_overlap": None,
                "top_chunk_overlap": None,
            },
            "primary_evidence": None,
            "sub_answers": [],
            "retrieval_debugging": [
                {
                    "attempt": 1,
                    "query": question,
                    "query_variants": [question],
                    "evaluation": evaluation,
                    "chunks": [],
                    "backend": "llamaindex",
                    "retrieval_time_seconds": round(retrieval_time_seconds, 2),
                    "planner_time_seconds": round(planner_time_seconds, 2),
                }
            ],
            "performance": {
                "retrieval_time_seconds": round(retrieval_time_seconds, 2),
                "query_rewrite_time_seconds": round(planner_time_seconds, 2),
                "answer_generation_time_seconds": 0.0,
                "answer_verification_time_seconds": 0.0,
                "total_pipeline_time_seconds": round(time.perf_counter() - pipeline_started, 2),
            },
        }

    answer_started = time.perf_counter()
    answer_payload = generate_answer_with_ollama(
        question,
        retrieved_chunks,
        model_name=llm_model_name,
        retrieval_query=question,
        max_context_chunks=simple_max_context_chunks,
        max_chars_per_chunk=simple_max_chars_per_chunk,
        temperature=answer_temperature,
        num_predict=simple_num_predict,
    )
    answer_generation_time_seconds = time.perf_counter() - answer_started

    return {
        "original_query": question,
        "rewritten_query": question,
        "canonical_query": "",
        "llm_query_rewrite": {"raw_response": "", "semantic_terms": []},
        "query_plan": query_plan,
        "query_intent": {"normalized_query": question, "tokens": question.split()},
        "generation_query": question,
        "query_variants_used": [question],
        "rewrite_used": False,
        "retrieval_attempts": 1,
        "retrieval_evaluation": evaluation,
        "answer_verification": {
            "verified": True,
            "repair_attempted": False,
            "unsupported_claims": [],
            "reason": "Verification not needed.",
        },
        "retrieved_chunks": retrieved_chunks,
        "final_answer": answer_payload["answer"],
        "fallback_triggered": answer_payload["answer"] == "I could not find this information in the uploaded documents.",
        "sources": answer_payload["sources"],
        "citations": answer_payload["citations"],
        "citations_text": answer_payload["citations_text"],
        "context_used": answer_payload["context_used"],
        "answer_strategy": "llamaindex_simple_workflow",
        "raw_generated_answer": answer_payload.get("raw_generated_answer"),
        "answer_quality": answer_payload.get("answer_quality"),
        "primary_evidence": answer_payload.get("primary_evidence"),
        "sub_answers": [],
        "retrieval_debugging": [
            {
                "attempt": 1,
                "query": question,
                "query_variants": [question],
                "evaluation": evaluation,
                "chunks": retrieved_chunks,
                "backend": "llamaindex",
                "retrieval_time_seconds": round(retrieval_time_seconds, 2),
                "planner_time_seconds": round(planner_time_seconds, 2),
            }
        ],
        "performance": {
            "retrieval_time_seconds": round(retrieval_time_seconds, 2),
            "query_rewrite_time_seconds": round(planner_time_seconds, 2),
            "answer_generation_time_seconds": round(answer_generation_time_seconds, 2),
            "answer_verification_time_seconds": 0.0,
            "total_pipeline_time_seconds": round(time.perf_counter() - pipeline_started, 2),
        },
    }


def _run_llamaindex_complex_answer(
    question: str,
    artifacts: PipelineArtifacts,
    *,
    llm_model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_chars_per_chunk: int,
    answer_temperature: float,
    answer_num_predict: int,
    reranker_model_name: str,
    query_plan: dict[str, Any],
    planner_time_seconds: float,
    pipeline_started: float,
) -> dict[str, Any]:
    """Run a complex LlamaIndex workflow using sub-query retrieval and answer composition."""
    sub_answers: list[dict[str, Any]] = []
    combined_citations: list[dict[str, Any]] = []
    combined_chunks: list[dict[str, Any]] = []
    retrieval_debugging: list[dict[str, Any]] = []
    retrieval_time_seconds = 0.0
    answer_generation_time_seconds = 0.0
    sub_max_context_chunks, sub_max_chars_per_chunk, sub_num_predict, combine_num_predict = _complex_subanswer_limits(
        max_context_chunks,
        max_chars_per_chunk,
        answer_num_predict,
    )
    route_type = str(query_plan.get("route_type", "general_complex"))

    for sub_query in query_plan.get("sub_queries", [])[:4]:
        sub_retrieval_started = time.perf_counter()
        sub_chunks = retrieve_llamaindex_chunks(
            sub_query,
            index=artifacts.llamaindex_index,
            chunks=artifacts.chunks,
            top_k=max(top_k, 6),
        )
        sub_chunks = _prepare_llamaindex_retrieved_chunks(
            sub_query,
            sub_chunks,
            reranker_model_name=reranker_model_name,
            top_k=max(top_k, 6),
        )
        sub_retrieval_time_seconds = time.perf_counter() - sub_retrieval_started
        retrieval_time_seconds += sub_retrieval_time_seconds
        sub_evaluation = evaluate_retrieval_quality(sub_query, sub_chunks)
        retrieval_debugging.append(
            {
                "attempt": len(retrieval_debugging) + 1,
                "query": sub_query,
                "query_variants": [sub_query],
                "evaluation": sub_evaluation,
                "chunks": sub_chunks,
                "backend": "llamaindex",
                "sub_query": True,
                "retrieval_time_seconds": round(sub_retrieval_time_seconds, 2),
            }
        )

        if not sub_chunks:
            sub_answers.append({"query": sub_query, "answer": FALLBACK_ANSWER})
            continue

        sub_answer_started = time.perf_counter()
        if route_type == "coordinated_interpretation":
            sub_answer_payload = generate_answer_with_ollama(
                sub_query,
                sub_chunks,
                model_name=llm_model_name,
                retrieval_query=sub_query,
                max_context_chunks=sub_max_context_chunks,
                max_chars_per_chunk=sub_max_chars_per_chunk,
                temperature=answer_temperature,
                num_predict=sub_num_predict,
            )
        elif route_type == "summary_topics":
            sub_answer_payload = extract_topic_facts_with_ollama(
                sub_query,
                sub_chunks,
                model_name=llm_model_name,
                max_context_chunks=max(sub_max_context_chunks, 3),
                max_chars_per_chunk=max(sub_max_chars_per_chunk, 1000),
                temperature=0.0,
                num_predict=sub_num_predict,
            )
        else:
            sub_answer_payload = summarize_topic_with_ollama(
                sub_query,
                sub_chunks,
                model_name=llm_model_name,
                max_context_chunks=sub_max_context_chunks,
                max_chars_per_chunk=sub_max_chars_per_chunk,
                temperature=answer_temperature,
                num_predict=sub_num_predict,
            )
        answer_generation_time_seconds += time.perf_counter() - sub_answer_started
        sub_answers.append(
            {
                "query": sub_query,
                "answer": sub_answer_payload["answer"],
                "facts": sub_answer_payload.get("facts", ""),
            }
        )
        combined_citations.extend(sub_answer_payload.get("citations", []))
        summary_chunks = sub_answer_payload.get("primary_evidence", {}).get("chunks") or sub_chunks[:2]
        combined_chunks.extend(summary_chunks[:2])

    combine_prompt = build_subanswer_combine_prompt(question, sub_answers)
    combine_started = time.perf_counter()
    combine_response = _generate_with_ollama(
        model_name=llm_model_name,
        prompt=combine_prompt,
        options={
            "temperature": answer_temperature,
            "num_predict": combine_num_predict,
            "top_p": 0.9,
        },
    )
    combine_time_seconds = time.perf_counter() - combine_started
    answer_generation_time_seconds += combine_time_seconds
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

    response_payload = attach_citations_to_response(final_answer, deduped_citations)
    return {
        "original_query": question,
        "rewritten_query": question,
        "canonical_query": "",
        "llm_query_rewrite": {"raw_response": "", "semantic_terms": []},
        "query_plan": query_plan,
        "query_intent": {"normalized_query": question, "tokens": question.split()},
        "generation_query": question,
        "query_variants_used": [question],
        "rewrite_used": False,
        "retrieval_attempts": len(query_plan.get("sub_queries", [])),
        "retrieval_evaluation": {
            "is_relevant": True,
            "has_partial_evidence": True,
            "average_score": 0.0,
            "average_keyword_overlap": 0.0,
            "average_reranker_score": None,
            "reason": "Complex LlamaIndex workflow used sub-query retrieval.",
            "chunk_count": len(combined_chunks),
        },
        "answer_verification": {
            "verified": True,
            "repair_attempted": False,
            "unsupported_claims": [],
            "reason": "Verification not needed.",
        },
        "retrieved_chunks": combined_chunks,
        "final_answer": response_payload["answer"],
        "fallback_triggered": response_payload["answer"] == FALLBACK_ANSWER,
        "sources": get_unique_sources(combined_chunks) if deduped_citations else [],
        "citations": response_payload["citations"],
        "citations_text": response_payload["citations_text"],
        "context_used": "\n\n".join(chunk.get("chunk_text", "") for chunk in combined_chunks[:max_context_chunks]),
        "answer_strategy": "llamaindex_complex_workflow",
        "raw_generated_answer": final_answer,
        "answer_quality": {
            "off_topic": False,
            "reason": "Answer composed from LlamaIndex sub-query answers.",
            "unsupported_claims": [],
            "query_overlap": None,
            "top_chunk_overlap": None,
        },
        "primary_evidence": None,
        "sub_answers": sub_answers,
        "retrieval_debugging": retrieval_debugging,
        "performance": {
            "retrieval_time_seconds": round(retrieval_time_seconds, 2),
            "query_rewrite_time_seconds": round(planner_time_seconds, 2),
            "answer_generation_time_seconds": round(answer_generation_time_seconds, 2),
            "answer_verification_time_seconds": 0.0,
            "total_pipeline_time_seconds": round(time.perf_counter() - pipeline_started, 2),
        },
    }


def _build_pipeline_artifacts(
    pages: list[dict[str, Any]],
    *,
    source_names: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> PipelineArtifacts:
    """Build in-memory retrieval artifacts from extracted pages."""
    processed_pages = preprocess_pages(pages)
    chunks = chunk_pages(processed_pages, chunk_size=chunk_size, overlap=chunk_overlap)

    llamaindex_summary: dict[str, Any] = {
        "available": llamaindex_is_available(),
        "built": False,
        "reason": "LlamaIndex backend selected.",
        "document_count": 0,
        "embedding_model": embedding_model_name,
    }
    llamaindex_index: Any = None
    llamaindex_documents: list[Any] = []
    if llamaindex_summary["available"]:
        llamaindex_index, llamaindex_documents, llamaindex_summary = build_llamaindex_index(
            chunks,
            embedding_model_name=embedding_model_name,
        )
    else:
        llamaindex_summary = {
            "available": False,
            "built": False,
            "reason": "LlamaIndex packages are not installed.",
            "document_count": 0,
            "embedding_model": embedding_model_name,
        }

    return PipelineArtifacts(
        pages=pages,
        processed_pages=processed_pages,
        chunks=chunks,
        extraction_summary=summarize_extraction(pages),
        preprocessing_summary=summarize_preprocessing(processed_pages),
        chunk_summary=summarize_chunks(chunks),
        vector_summary=summarize_vector_store(chunks, model_name=embedding_model_name),
        vector_index=None,
        vector_metadata=[],
        lexical_index={},
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_names=source_names,
        rag_backend=RAG_BACKEND,
        llamaindex_summary=llamaindex_summary,
        llamaindex_index=llamaindex_index,
        llamaindex_documents=llamaindex_documents,
    )
