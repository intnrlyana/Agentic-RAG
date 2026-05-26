from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agentic_rag import DEFAULT_OLLAMA_MODEL, OllamaServiceError, check_ollama_connection
from src.api_service import (
    PipelineArtifacts,
    process_local_documents,
    process_uploaded_documents,
    run_agentic_answer,
    run_semantic_retrieval,
)


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class ProcessLocalRequest(BaseModel):
    data_dir: str = "data"
    chunk_size: int = 900
    chunk_overlap: int = 180
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"


class ProcessSummaryResponse(BaseModel):
    source_names: list[str]
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    rag_backend: str
    extraction_summary: dict[str, Any]
    preprocessing_summary: dict[str, Any]
    chunk_summary: dict[str, Any]
    vector_summary: dict[str, Any]
    llamaindex_summary: dict[str, Any]
    sample_pages: list[dict[str, Any]]
    sample_processed_pages: list[dict[str, Any]]
    sample_chunks: list[dict[str, Any]]


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class AskRequest(BaseModel):
    question: str
    llm_model_name: str = DEFAULT_OLLAMA_MODEL
    top_k: int = 10
    max_context_chunks: int = 4
    max_chars_per_chunk: int = 1200
    answer_temperature: float = 0.1
    answer_num_predict: int = 250
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HealthResponse(BaseModel):
    status: str
    documents_loaded: bool


class OllamaStatusResponse(BaseModel):
    ok: bool
    message: str


class AppState:
    """Minimal in-memory API state."""

    def __init__(self) -> None:
        self.artifacts: PipelineArtifacts | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Agentic RAG Document Assistant API",
    version="1.0.0",
    summary="REST backend for document processing, retrieval, and agentic RAG answering.",
    lifespan=lifespan,
)
app.state.runtime = AppState()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents_loaded=app.state.runtime.artifacts is not None,
    )


@app.get("/ollama/status", response_model=OllamaStatusResponse)
def ollama_status(model_name: str = DEFAULT_OLLAMA_MODEL) -> OllamaStatusResponse:
    ok, message = check_ollama_connection(model_name)
    return OllamaStatusResponse(ok=ok, message=message)


@app.get("/documents/status", response_model=ProcessSummaryResponse)
def documents_status() -> ProcessSummaryResponse:
    artifacts = _require_artifacts()
    return _build_process_summary_response(artifacts)


@app.post("/documents/process/local", response_model=ProcessSummaryResponse)
def process_local(request: ProcessLocalRequest) -> ProcessSummaryResponse:
    artifacts = process_local_documents(
        data_dir=request.data_dir,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        embedding_model_name=request.embedding_model_name,
    )
    app.state.runtime.artifacts = artifacts
    return _build_process_summary_response(artifacts)


@app.post("/documents/process/upload", response_model=ProcessSummaryResponse)
async def process_upload(
    files: list[UploadFile] = File(...),
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
) -> ProcessSummaryResponse:
    uploaded_files: list[tuple[str, bytes]] = []
    for file in files:
        file_bytes = await file.read()
        uploaded_files.append((file.filename or "uploaded_document.pdf", file_bytes))

    artifacts = process_uploaded_documents(
        uploaded_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )
    app.state.runtime.artifacts = artifacts
    return _build_process_summary_response(artifacts)


@app.post("/retrieve")
def retrieve(request: RetrievalRequest) -> dict[str, Any]:
    artifacts = _require_artifacts()
    results = run_semantic_retrieval(
        request.query,
        artifacts,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        reranker_model_name=request.reranker_model_name,
    )
    return {
        "query": request.query,
        "top_k": request.top_k,
        "use_reranker": request.use_reranker,
        "results": results,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    artifacts = _require_artifacts()
    request_id = uuid4().hex[:8]
    _log_ask_request(request_id, request)
    try:
        result = run_agentic_answer(
            request.question,
            artifacts,
            llm_model_name=request.llm_model_name,
            top_k=request.top_k,
            max_context_chunks=request.max_context_chunks,
            max_chars_per_chunk=request.max_chars_per_chunk,
            answer_temperature=request.answer_temperature,
            answer_num_predict=request.answer_num_predict,
            use_reranker=request.use_reranker,
            reranker_model_name=request.reranker_model_name,
        )
        _log_ask_result(request_id, request, result)
        return result
    except OllamaServiceError as exc:
        logger.exception(
            "[ASK %s] OllamaServiceError | question=%r | model=%s | detail=%s",
            request_id,
            request.question,
            request.llm_model_name,
            exc.message,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _require_artifacts() -> PipelineArtifacts:
    artifacts = app.state.runtime.artifacts
    if artifacts is None:
        raise HTTPException(
            status_code=409,
            detail="No processed documents are loaded. Process local or uploaded PDFs first.",
        )
    return artifacts


def _build_process_summary_response(artifacts: PipelineArtifacts) -> ProcessSummaryResponse:
    return ProcessSummaryResponse(
        source_names=artifacts.source_names,
        embedding_model_name=artifacts.embedding_model_name,
        chunk_size=artifacts.chunk_size,
        chunk_overlap=artifacts.chunk_overlap,
        rag_backend=artifacts.rag_backend,
        extraction_summary=artifacts.extraction_summary,
        preprocessing_summary=artifacts.preprocessing_summary,
        chunk_summary=artifacts.chunk_summary,
        vector_summary=artifacts.vector_summary,
        llamaindex_summary=artifacts.llamaindex_summary,
        sample_pages=artifacts.pages[:3],
        sample_processed_pages=artifacts.processed_pages[:3],
        sample_chunks=artifacts.chunks[:5],
    )


def _log_ask_request(request_id: str, request: AskRequest) -> None:
    """Print a concise request header for each /ask call."""
    logger.info(
        "[ASK %s] START | question=%r | model=%s | top_k=%s | max_context_chunks=%s | max_chars_per_chunk=%s | reranker=%s",
        request_id,
        request.question,
        request.llm_model_name,
        request.top_k,
        request.max_context_chunks,
        request.max_chars_per_chunk,
        request.use_reranker,
    )


def _log_ask_result(request_id: str, request: AskRequest, result: dict[str, Any]) -> None:
    """Print the retrieval/answer flow and final answer for each /ask call."""
    flow_summary = {
        "original_query": result.get("original_query"),
        "rewritten_query": result.get("rewritten_query"),
                "canonical_query": result.get("canonical_query"),
                "llm_query_rewrite": result.get("llm_query_rewrite"),
                "query_plan": result.get("query_plan"),
                "generation_query": result.get("generation_query"),
                "query_intent": result.get("query_intent"),
                "query_variants_used": result.get("query_variants_used"),
        "retrieval_attempts": result.get("retrieval_attempts"),
        "retrieval_evaluation": result.get("retrieval_evaluation"),
        "answer_strategy": result.get("answer_strategy"),
        "fallback_triggered": result.get("fallback_triggered"),
        "performance": result.get("performance"),
    }
    top_chunks = [
        {
            "rank": chunk.get("rank"),
            "source": chunk.get("source"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "score": round(float(chunk.get("score", 0.0)), 4),
            "faiss_score": round(float(chunk.get("faiss_score", 0.0)), 4),
            "keyword_overlap": round(float(chunk.get("keyword_overlap", 0.0)), 4),
            "reranker_score": (
                round(float(chunk.get("reranker_score", 0.0)), 4)
                if chunk.get("reranker_score") is not None
                else None
            ),
            "matched_queries": chunk.get("matched_queries", []),
            "preview": _compact_preview(chunk.get("chunk_text", "")),
        }
        for chunk in result.get("retrieved_chunks", [])[:3]
    ]
    retrieval_attempts = [
        {
            "attempt": attempt.get("attempt"),
            "query": attempt.get("query"),
            "query_variants": attempt.get("query_variants", []),
            "evaluation": attempt.get("evaluation"),
            "top_chunks": [
                {
                    "rank": chunk.get("rank"),
                    "source": chunk.get("source"),
                    "page_number": chunk.get("page_number"),
                    "chunk_id": chunk.get("chunk_id"),
                    "score": round(float(chunk.get("score", 0.0)), 4),
                    "preview": _compact_preview(chunk.get("chunk_text", "")),
                }
                for chunk in attempt.get("chunks", [])[:3]
            ],
        }
        for attempt in result.get("retrieval_debugging", [])
    ]
    answer_summary = {
        "raw_generated_answer": result.get("raw_generated_answer"),
        "final_answer": result.get("final_answer"),
        "primary_evidence": result.get("primary_evidence"),
        "answer_quality": result.get("answer_quality"),
        "sub_answers": result.get("sub_answers", []),
        "citations": [
            {
                "citation_id": citation.get("citation_id"),
                "source": citation.get("source"),
                "page_number": citation.get("page_number"),
                "chunk_id": citation.get("chunk_id"),
            }
            for citation in result.get("citations", [])
        ],
        "verification": result.get("answer_verification"),
    }

    logger.info(
        "[ASK %s] FLOW\n%s",
        request_id,
        json.dumps(
            _json_safe(
                {
                "request": {
                    "question": request.question,
                    "model": request.llm_model_name,
                },
                "flow": flow_summary,
                "retrieval_attempts": retrieval_attempts,
                "top_retrieved_chunks": top_chunks,
                "answer": answer_summary,
                }
            ),
            indent=2,
            ensure_ascii=True,
        ),
    )
    logger.info("[ASK %s] END", request_id)


def _compact_preview(text: str, limit: int = 220) -> str:
    """Return a one-line preview for terminal logs."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _json_safe(value: Any) -> Any:
    """Convert nested result payloads into JSON-serializable terminal log data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
