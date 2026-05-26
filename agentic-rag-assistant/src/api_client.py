"""HTTP client helpers for the Streamlit frontend to talk to the FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests


class ApiClientError(RuntimeError):
    """Raised when the FastAPI backend returns an error or is unreachable."""


def get_api_health(base_url: str) -> dict[str, Any]:
    """Check whether the FastAPI backend is up."""
    return _request_json("GET", f"{_normalize_base_url(base_url)}/health")


def get_ollama_status(base_url: str, model_name: str) -> dict[str, Any]:
    """Check LLM provider status through the backend."""
    return _request_json(
        "GET",
        f"{_normalize_base_url(base_url)}/ollama/status",
        params={"model_name": model_name},
    )


def process_local_documents(
    base_url: str,
    *,
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> dict[str, Any]:
    """Ask the backend to process local PDFs from the data directory."""
    return _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/documents/process/local",
        json={
            "data_dir": data_dir,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model_name": embedding_model_name,
        },
    )


def process_uploaded_documents(
    base_url: str,
    uploaded_files: list[Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> dict[str, Any]:
    """Upload PDFs to the backend and trigger processing."""
    files = []
    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.read(),
                    "application/pdf",
                ),
            )
        )
    return _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/documents/process/upload",
        params={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model_name": embedding_model_name,
        },
        files=files,
    )


def retrieve_chunks(
    base_url: str,
    *,
    query: str,
    top_k: int,
    use_reranker: bool,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, Any]:
    """Call the backend retrieval endpoint."""
    return _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/retrieve",
        json={
            "query": query,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "reranker_model_name": reranker_model_name,
        },
    )


def ask_agentic_rag(
    base_url: str,
    *,
    question: str,
    llm_model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_chars_per_chunk: int,
    answer_temperature: float,
    answer_num_predict: int,
    use_reranker: bool,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, Any]:
    """Call the backend agentic RAG endpoint."""
    return _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/ask",
        json={
            "question": question,
            "llm_model_name": llm_model_name,
            "top_k": top_k,
            "max_context_chunks": max_context_chunks,
            "max_chars_per_chunk": max_chars_per_chunk,
            "answer_temperature": answer_temperature,
            "answer_num_predict": answer_num_predict,
            "use_reranker": use_reranker,
            "reranker_model_name": reranker_model_name,
        },
    )


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """Execute an HTTP request and return JSON, raising a clean error on failure."""
    try:
        response = requests.request(method, url, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text
        raise ApiClientError(detail or str(exc)) from exc
    except requests.RequestException as exc:
        raise ApiClientError(f"Could not reach FastAPI backend at {url}.") from exc


def _normalize_base_url(base_url: str) -> str:
    """Normalize a base URL so endpoint joining is predictable."""
    return base_url.rstrip("/")
