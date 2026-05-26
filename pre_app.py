from __future__ import annotations

import os
from pathlib import Path
import time

import streamlit as st
from dotenv import load_dotenv

from src.api_client import (
    ApiClientError,
    ask_agentic_rag,
    get_api_health,
    get_ollama_status,
    process_local_documents as api_process_local_documents,
    process_uploaded_documents as api_process_uploaded_documents,
    retrieve_chunks,
)
from src.evaluation import get_default_test_cases, score_test_result, summarize_test_results


DATA_DIR = Path("data")
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
if not (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env.example", override=False)


def _initialize_session_state() -> None:
    """Set default session keys used across processing and demo interactions."""
    defaults = {
        "pages": [],
        "processed_pages": [],
        "chunks": [],
        "documents_ready": False,
        "extraction_summary": None,
        "preprocessing_summary": None,
        "chunk_summary": None,
        "vector_summary": None,
        "llamaindex_summary": None,
        "rag_backend": None,
        "selected_source_names": [],
        "attempted_pdf_count": 0,
        "embedding_model_name": None,
        "retrieval_results": [],
        "ollama_status": None,
        "api_status": None,
        "agentic_result": None,
        "evaluation_results": [],
        "evaluation_summary": None,
        "last_processed_message": "",
        "processing_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_app_state() -> None:
    """Clear derived app state while keeping Streamlit itself stable."""
    keys_to_reset = [
        "pages",
        "processed_pages",
        "chunks",
        "documents_ready",
        "extraction_summary",
        "preprocessing_summary",
        "chunk_summary",
        "vector_summary",
        "llamaindex_summary",
        "rag_backend",
        "selected_source_names",
        "attempted_pdf_count",
        "embedding_model_name",
        "retrieval_results",
        "ollama_status",
        "api_status",
        "agentic_result",
        "evaluation_results",
        "evaluation_summary",
        "last_processed_message",
        "processing_signature",
    ]
    for key in keys_to_reset:
        st.session_state[key] = [] if key in {
            "pages",
            "processed_pages",
            "chunks",
            "selected_source_names",
            "retrieval_results",
            "evaluation_results",
        } else None

    st.session_state["attempted_pdf_count"] = 0
    st.session_state["documents_ready"] = False
    st.session_state["last_processed_message"] = ""


def _truncate_text(text: str, limit: int = 500) -> str:
    """Return a compact preview for large text blocks."""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _show_ollama_status() -> None:
    """Render the latest LLM connection status if available."""
    status = st.session_state.ollama_status
    if not status:
        return
    if status["ok"]:
        st.success(status["message"])
    else:
        st.warning(status["message"])


def _show_api_status() -> None:
    """Render the latest FastAPI backend status if available."""
    status = st.session_state.api_status
    if not status:
        return
    if status["ok"]:
        st.success(status["message"])
    else:
        st.warning(status["message"])


def _render_summary_metrics() -> None:
    """Render document-processing summary blocks."""
    pages = st.session_state.pages
    summary = st.session_state.extraction_summary
    preprocessing_summary = st.session_state.preprocessing_summary
    chunk_summary = st.session_state.chunk_summary
    vector_summary = st.session_state.vector_summary
    llamaindex_summary = st.session_state.llamaindex_summary
    rag_backend = st.session_state.rag_backend

    if not pages or not summary:
        st.info("No processed documents yet. Use the sidebar to load PDFs and click `Process Documents`.")
        return

    failed_or_empty_sources = sorted(
        set(st.session_state.selected_source_names) - set(summary["source_filenames"])
    )

    st.markdown("**Extraction Summary**")
    cols = st.columns(4)
    cols[0].metric("PDFs selected", st.session_state.attempted_pdf_count)
    cols[1].metric("PDFs with pages", summary["pdf_count"])
    cols[2].metric("Pages extracted", summary["total_pages_extracted"])
    cols[3].metric("Empty pages", summary["empty_or_skipped_pages"])
    st.caption("Source files: " + ", ".join(st.session_state.selected_source_names))
    if failed_or_empty_sources:
        st.warning(
            "Some PDFs returned no readable pages: " + ", ".join(failed_or_empty_sources)
        )

    if preprocessing_summary:
        st.markdown("**Preprocessing Summary**")
        cols = st.columns(5)
        cols[0].metric("Total pages", preprocessing_summary["total_pages"])
        cols[1].metric("Retained", preprocessing_summary["retained_pages"])
        cols[2].metric("Skipped", preprocessing_summary["skipped_pages"])
        cols[3].metric("Original chars", preprocessing_summary["total_original_characters"])
        cols[4].metric("Cleaned chars", preprocessing_summary["total_cleaned_characters"])
        if preprocessing_summary["skipped_by_reason"]:
            st.caption(
                "Skipped by reason: "
                + ", ".join(
                    f"{reason}: {count}"
                    for reason, count in preprocessing_summary["skipped_by_reason"].items()
                )
            )

    if chunk_summary:
        st.markdown("**Chunking Summary**")
        cols = st.columns(5)
        cols[0].metric("Total chunks", chunk_summary["total_chunks"])
        cols[1].metric("Documents chunked", chunk_summary["total_documents"])
        cols[2].metric("Avg chunk length", chunk_summary["average_chunk_length"])
        cols[3].metric("Min chunk length", chunk_summary["min_chunk_length"])
        cols[4].metric("Max chunk length", chunk_summary["max_chunk_length"])

    if vector_summary:
        st.markdown("**Vector Store Summary**")
        cols = st.columns(4)
        cols[0].metric("Total vectors", vector_summary["total_vectors"])
        cols[1].metric("Documents indexed", vector_summary["total_documents"])
        cols[2].metric("Embedding model", vector_summary["embedding_model"])
        cols[3].metric("Filtered chunks", vector_summary["filtered_low_information_chunks"])

    if llamaindex_summary is not None:
        st.markdown("**Backend Summary**")
        cols = st.columns(4)
        cols[0].metric("RAG backend", rag_backend or "llamaindex")
        cols[1].metric("LlamaIndex available", "Yes" if llamaindex_summary.get("available") else "No")
        cols[2].metric("LlamaIndex built", "Yes" if llamaindex_summary.get("built") else "No")
        cols[3].metric("LlamaIndex docs", llamaindex_summary.get("document_count", 0))
        if llamaindex_summary.get("reason"):
            st.caption(f"LlamaIndex: {llamaindex_summary.get('reason')}")


def _render_citations(citations: list[dict], key_prefix: str = "citations") -> None:
    """Render citations with expandable chunk previews."""
    if not citations:
        return
    st.markdown("**Citations**")
    citation_lines = [
        f"[{citation['citation_id']}] {citation['source']}, Page {citation['page_number']}"
        for citation in citations
    ]
    st.code("\n".join(citation_lines), language="text")
    with st.expander("Citation Details", expanded=False):
        for citation in citations:
            st.markdown(
                f"**[{citation['citation_id']}] {citation['source']}, Page {citation['page_number']}**  \n"
                f"Chunk: {citation['chunk_id']} | Score: {citation['score']:.4f}"
            )
            st.text_area(
                label=f"citation-{citation['chunk_id']}",
                value=_truncate_text(citation.get("chunk_text", ""), 700),
                height=120,
                disabled=True,
                label_visibility="collapsed",
                key=f"{key_prefix}-citation-{citation['citation_id']}-{citation['chunk_id']}",
            )


def _build_processing_signature(
    backend_url: str,
    source_mode: str,
    selected_source_names: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> tuple:
    """Build a lightweight processing signature for session-level reuse."""
    return (
        backend_url,
        source_mode,
        tuple(selected_source_names),
        chunk_size,
        chunk_overlap,
        embedding_model_name,
    )


def _apply_process_response(response: dict, attempted_pdf_count: int) -> None:
    """Map backend processing output into the UI session state."""
    st.session_state.pages = response.get("sample_pages", [])
    st.session_state.processed_pages = response.get("sample_processed_pages", [])
    st.session_state.chunks = response.get("sample_chunks", [])
    st.session_state.documents_ready = True
    st.session_state.extraction_summary = response.get("extraction_summary")
    st.session_state.preprocessing_summary = response.get("preprocessing_summary")
    st.session_state.chunk_summary = response.get("chunk_summary")
    st.session_state.vector_summary = response.get("vector_summary")
    st.session_state.llamaindex_summary = response.get("llamaindex_summary")
    st.session_state.rag_backend = response.get("rag_backend")
    st.session_state.selected_source_names = response.get("source_names", [])
    st.session_state.attempted_pdf_count = attempted_pdf_count
    st.session_state.embedding_model_name = response.get("embedding_model_name")
    st.session_state.retrieval_results = []
    st.session_state.agentic_result = None
    st.session_state.evaluation_results = []
    st.session_state.evaluation_summary = None
    st.session_state.last_processed_message = (
        f"Processed {response['extraction_summary']['pdf_count']} PDF(s) and built "
        f"{response['vector_summary']['total_vectors']} vectors via FastAPI."
    )


def _run_api_test_case(
    base_url: str,
    test_case: dict,
    *,
    ollama_model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_context_chars: int,
    answer_temperature: float,
    answer_num_predict: int,
    use_reranker: bool,
) -> dict:
    """Run one predefined evaluation case through the FastAPI backend."""
    started_at = time.perf_counter()
    agentic_result = ask_agentic_rag(
        base_url,
        question=test_case["question"],
        llm_model_name=ollama_model_name,
        top_k=top_k,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_context_chars,
        answer_temperature=answer_temperature,
        answer_num_predict=answer_num_predict,
        use_reranker=use_reranker,
    )
    latency_seconds = round(time.perf_counter() - started_at, 2)
    result = {
        "test_name": test_case["test_name"],
        "question": test_case["question"],
        "expected_behavior": test_case["expected_behavior"],
        "test_type": test_case["test_type"],
        "expected_facts": test_case.get("expected_facts", []),
        "expected_pages": test_case.get("expected_pages", []),
        "answer": agentic_result["final_answer"],
        "citations": agentic_result["citations"],
        "retrieval_attempts": agentic_result["retrieval_attempts"],
        "fallback_triggered": agentic_result["fallback_triggered"],
        "retrieval_evaluation": agentic_result["retrieval_evaluation"],
        "latency_seconds": latency_seconds,
        "notes": "Golden fact check available below.",
        "agentic_result": agentic_result,
    }
    result["metrics"] = score_test_result(test_case, agentic_result)
    return result


_initialize_session_state()

st.set_page_config(
    page_title="Agentic RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("Agentic RAG Document Assistant")
st.caption(
    "Phase 10.5: optimized Streamlit demo with retrieval-first agent flow, "
    "trimmed context, and lightweight performance tracking."
)

with st.sidebar:
    st.header("Controls")
    backend_url = st.text_input(
        "FastAPI backend URL",
        value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
    )
    source_mode = st.radio(
        "Document source",
        options=["Upload PDFs", "Load from data/ folder"],
    )

    uploaded_files = []
    selected_source_names: list[str] = []
    if source_mode == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload one or more PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        selected_source_names = [uploaded_file.name for uploaded_file in uploaded_files]
        if selected_source_names:
            st.info(f"{len(selected_source_names)} uploaded PDF(s) selected.")
        else:
            st.info("No uploaded PDFs selected.")
    else:
        local_pdfs = sorted(path.name for path in DATA_DIR.glob("*.pdf"))
        selected_source_names = local_pdfs
        if local_pdfs:
            st.info(f"Found {len(local_pdfs)} PDF(s) in `data/`.")
            st.write("\n".join(f"- {name}" for name in local_pdfs))
        else:
            st.warning("No PDF files found in `data/`.")

    st.divider()
    chunk_size = st.slider("Chunk size", min_value=300, max_value=1600, value=900, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=180, step=20)
    top_k = st.slider("Initial retrieval depth", min_value=3, max_value=15, value=10, step=1)
    max_context_chunks = st.slider("Final context chunks", min_value=1, max_value=6, value=4, step=1)
    max_context_chars = st.slider("Max chars per context chunk", min_value=300, max_value=2000, value=1200, step=100)
    embedding_model_name = st.selectbox(
        "Embedding model",
        options=["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"],
        index=0,
    )
    use_reranker = True
    st.caption("Cross-encoder reranker is always enabled for the final system.")
    ollama_model_name = st.text_input(
        "LLM model",
        value=os.getenv("GROQ_MODEL_NAME", os.getenv("LLM_MODEL_NAME", os.getenv("OLLAMA_MODEL_NAME", "llama-3.3-70b-versatile"))),
    )
    answer_num_predict = st.slider("Max output tokens", min_value=80, max_value=500, value=250, step=10)
    answer_temperature = st.slider("Answer temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    st.divider()
    check_backend_clicked = st.button("Check FastAPI Backend", width="stretch")
    check_ollama_clicked = st.button("Check LLM Connection", width="stretch")
    process_clicked = st.button("Process Documents", type="primary", width="stretch")
    reset_clicked = st.button("Reset / Clear Session", width="stretch")

if reset_clicked:
    _reset_app_state()
    st.rerun()

if check_ollama_clicked:
    with st.spinner("Checking LLM connection..."):
        try:
            status = get_ollama_status(backend_url, ollama_model_name)
            st.session_state.ollama_status = status
        except ApiClientError as exc:
            st.session_state.ollama_status = {"ok": False, "message": str(exc)}

if check_backend_clicked:
    with st.spinner("Checking FastAPI backend..."):
        try:
            health = get_api_health(backend_url)
            st.session_state.api_status = {
                "ok": True,
                "message": (
                    f"FastAPI backend is running. Documents loaded: {health.get('documents_loaded', False)}"
                ),
            }
        except ApiClientError as exc:
            st.session_state.api_status = {"ok": False, "message": str(exc)}

if process_clicked:
    attempted_pdf_count = len(selected_source_names)
    processing_signature = _build_processing_signature(
        backend_url,
        source_mode,
        selected_source_names,
        chunk_size,
        chunk_overlap,
        embedding_model_name,
    )

    if (
        st.session_state.processing_signature == processing_signature
        and st.session_state.documents_ready
    ):
        st.info("Using the existing processed session state. No rebuild was needed.")
    else:
        try:
            with st.spinner("Sending documents to FastAPI for extraction, chunking, and indexing..."):
                if source_mode == "Upload PDFs":
                    if not uploaded_files:
                        st.warning("Select at least one PDF file before processing.")
                        response = None
                    else:
                        response = api_process_uploaded_documents(
                            backend_url,
                            uploaded_files,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            embedding_model_name=embedding_model_name,
                        )
                else:
                    if not selected_source_names:
                        st.warning("Add PDF files to the `data/` folder before processing.")
                        response = None
                    else:
                        response = api_process_local_documents(
                            backend_url,
                            data_dir=str(DATA_DIR),
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            embedding_model_name=embedding_model_name,
                        )
            if response:
                _apply_process_response(response, attempted_pdf_count=attempted_pdf_count)
                st.session_state.processing_signature = processing_signature
                st.session_state.api_status = {"ok": True, "message": "FastAPI backend processed documents successfully."}
                st.success(st.session_state.last_processed_message)
        except ApiClientError as exc:
            st.error(str(exc))
            st.session_state.api_status = {"ok": False, "message": str(exc)}

pages = st.session_state.pages
processed_pages = st.session_state.processed_pages
chunks = st.session_state.chunks

if not st.session_state.documents_ready:
    st.warning("No documents are loaded yet. Use the sidebar to upload PDFs or load them from `data/`.")
else:
    st.success(st.session_state.last_processed_message or "Documents are ready.")

_show_api_status()
_show_ollama_status()

tab_processing, tab_retrieval, tab_agentic, tab_evaluation = st.tabs(
    [
        "Document Processing",
        "Semantic Retrieval Test",
        "Agentic RAG Chat",
        "Testing & Evaluation",
    ]
)

with tab_processing:
    _render_summary_metrics()

    if pages:
        sample_raw_pages = pages[:3]
        sample_processed_pages = processed_pages[:3]
        sample_chunks = chunks[:5]

        with st.expander("Sample Extracted Text", expanded=False):
            for page in sample_raw_pages:
                st.markdown(f"**{page['source']} | Page {page['page_number']}**")
                st.text_area(
                    label=f"raw-page-{page['source']}-{page['page_number']}",
                    value=_truncate_text(page.get("text", "") or "[No text extracted]", 900),
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )

        with st.expander("Sample Cleaned Text", expanded=False):
            for page in sample_processed_pages:
                st.markdown(f"**{page['source']} | Page {page['page_number']}**")
                st.caption(
                    f"Original length: {page['original_text_length']} | "
                    f"Cleaned length: {page['cleaned_text_length']} | "
                    f"Skipped: {page['was_skipped']}"
                )
                st.text_area(
                    label=f"cleaned-page-{page['source']}-{page['page_number']}",
                    value=_truncate_text(page.get("text", "") or "[No retained text]", 900),
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )

        with st.expander("Sample Chunks", expanded=False):
            for chunk in sample_chunks:
                st.markdown(
                    f"**{chunk['chunk_id']}**  \n"
                    f"{chunk['source']} | Page {chunk['page_number']} | "
                    f"Length: {chunk['chunk_length']}"
                )
                st.text_area(
                    label=f"sample-chunk-{chunk['chunk_id']}",
                    value=_truncate_text(chunk["chunk_text"], 800),
                    height=130,
                    disabled=True,
                    label_visibility="collapsed",
                )

with tab_retrieval:
    st.markdown("**Semantic Retrieval Test**")
    if not st.session_state.documents_ready:
        st.warning("Backend index is not ready. Process documents first.")
    else:
        retrieval_query = st.text_input("Enter a retrieval query", key="tab_retrieval_query")
        if st.button("Run Retrieval", key="tab_retrieval_button"):
            if not retrieval_query.strip():
                st.warning("Enter a query before running retrieval.")
            else:
                try:
                    with st.spinner("Calling backend retrieval..."):
                        retrieval_response = retrieve_chunks(
                            backend_url,
                            query=retrieval_query,
                            top_k=top_k,
                            use_reranker=use_reranker,
                        )
                    st.session_state.retrieval_results = retrieval_response["results"]
                except ApiClientError as exc:
                    st.error(str(exc))
                    st.session_state.api_status = {"ok": False, "message": str(exc)}

        if st.session_state.retrieval_results:
            for result in st.session_state.retrieval_results:
                with st.expander(
                    f"Rank {result['rank']} | Score {result['score']:.4f} | {result['chunk_id']}",
                    expanded=result["rank"] == 1,
                ):
                    st.write(
                        f"Source: {result['source']} | "
                        f"Page: {result['page_number']} | "
                        f"Type: {result['source_type']}"
                    )
                    st.caption(
                        f"FAISS score: {result['faiss_score']:.4f} | "
                        f"Keyword overlap: {result['keyword_overlap']:.4f}"
                    )
                    if result.get("reranker_score") is not None:
                        st.caption(f"Reranker score: {result['reranker_score']:.4f}")
                    st.text_area(
                        label=f"retrieval-result-{result['chunk_id']}",
                        value=_truncate_text(result["chunk_text"], 900),
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )

with tab_agentic:
    st.markdown("**Agentic RAG Chat**")
    if not st.session_state.documents_ready:
        st.warning("Backend index is not ready. Process documents first.")
    else:
        agentic_query = st.text_input("Ask a question about the documents", key="tab_agentic_query")
        if st.button("Run Agentic RAG", key="tab_agentic_button"):
            if not agentic_query.strip():
                st.warning("Enter a question before running Agentic RAG.")
            else:
                try:
                    status = get_ollama_status(backend_url, ollama_model_name)
                    st.session_state.ollama_status = status
                    if not status["ok"]:
                        st.error(status["message"])
                    else:
                        with st.spinner("Calling backend agentic RAG pipeline..."):
                            st.session_state.agentic_result = ask_agentic_rag(
                                backend_url,
                                question=agentic_query,
                                llm_model_name=ollama_model_name,
                                top_k=top_k,
                                max_context_chunks=max_context_chunks,
                                max_chars_per_chunk=max_context_chars,
                                answer_temperature=answer_temperature,
                                answer_num_predict=answer_num_predict,
                                use_reranker=use_reranker,
                            )
                except ApiClientError as exc:
                    st.error(str(exc))
                    st.session_state.api_status = {"ok": False, "message": str(exc)}

        if st.session_state.agentic_result:
            agentic_result = st.session_state.agentic_result
            st.markdown("**Final Answer**")
            st.write(agentic_result["final_answer"])

            if not agentic_result["fallback_triggered"]:
                _render_citations(agentic_result["citations"], key_prefix="agentic")

            with st.expander("Agentic Reasoning", expanded=False):
                st.write(f"Original query: {agentic_result['original_query']}")
                st.write(f"Rewritten query: {agentic_result['rewritten_query']}")
                st.write(f"Canonical query: {agentic_result.get('canonical_query', '')}")
                st.write(f"LLM query rewrite: {agentic_result.get('llm_query_rewrite', {})}")
                st.write(f"Query plan: {agentic_result.get('query_plan', {})}")
                st.write(f"Answer strategy: {agentic_result.get('answer_strategy', 'unknown')}")
                st.write(f"Query intent: {agentic_result.get('query_intent', {})}")
                if agentic_result.get("primary_evidence") is not None:
                    st.write(f"Primary evidence: {agentic_result.get('primary_evidence')}")
                if agentic_result.get("sub_answers"):
                    st.write(f"Sub-answers: {agentic_result.get('sub_answers')}")
                if agentic_result.get("raw_generated_answer"):
                    st.write(f"Raw generated answer: {agentic_result.get('raw_generated_answer')}")
                if agentic_result.get("answer_quality") is not None:
                    st.write(f"Answer quality check: {agentic_result.get('answer_quality')}")
                st.write(f"Retrieval attempts: {agentic_result['retrieval_attempts']}")
                st.write(f"Retrieval evaluation: {agentic_result['retrieval_evaluation']}")
                st.write(f"Fallback triggered: {agentic_result['fallback_triggered']}")
                st.write(f"Rewrite used: {agentic_result['rewrite_used']}")

            with st.expander("Retrieval Debugging", expanded=False):
                st.write(f"Original query: {agentic_result['original_query']}")
                st.write(f"Rewritten query: {agentic_result['rewritten_query']}")
                for attempt in agentic_result.get("retrieval_debugging", []):
                    st.markdown(f"**Attempt {attempt['attempt']} | Query: {attempt['query']}**")
                    st.write(f"Evaluation: {attempt['evaluation']}")
                    for chunk in attempt["chunks"]:
                        st.markdown(
                            f"Rank {chunk['rank']} | {chunk['source']} | Page {chunk['page_number']} | "
                            f"FAISS {chunk['faiss_score']:.4f}"
                        )
                        if chunk.get("reranker_score") is not None:
                            st.caption(
                                f"Reranker score: {chunk['reranker_score']:.4f} | "
                                f"Keyword overlap: {chunk['keyword_overlap']:.4f}"
                            )
                        else:
                            st.caption(f"Keyword overlap: {chunk['keyword_overlap']:.4f}")
                        st.text_area(
                            label=f"debug-{attempt['attempt']}-{chunk['chunk_id']}",
                            value=_truncate_text(chunk["chunk_text"], 700),
                            height=110,
                            disabled=True,
                            label_visibility="collapsed",
                        )

            if agentic_result["retrieved_chunks"]:
                with st.expander("Retrieved Chunks", expanded=False):
                    for chunk in agentic_result["retrieved_chunks"]:
                        st.markdown(
                            f"**Rank {chunk['rank']} | Score {chunk['score']:.4f} | {chunk['chunk_id']}**  \n"
                            f"{chunk['source']} | Page {chunk['page_number']}"
                        )
                        caption_parts = [f"FAISS score: {chunk['faiss_score']:.4f}"]
                        caption_parts.append(f"Keyword overlap: {chunk['keyword_overlap']:.4f}")
                        if chunk.get("reranker_score") is not None:
                            caption_parts.append(f"Reranker score: {chunk['reranker_score']:.4f}")
                        st.caption(" | ".join(caption_parts))
                        st.text_area(
                            label=f"agentic-chunk-{chunk['chunk_id']}",
                            value=_truncate_text(chunk["chunk_text"], 900),
                            height=130,
                            disabled=True,
                            label_visibility="collapsed",
                        )

            if agentic_result["context_used"]:
                with st.expander("Context Used", expanded=False):
                    st.text_area(
                        label="agentic-context-used",
                        value=_truncate_text(agentic_result["context_used"], 2500),
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            with st.expander("Performance Details", expanded=False):
                performance = agentic_result["performance"]
                st.write(f"Retrieval time: {performance['retrieval_time_seconds']} seconds")
                st.write(f"Query rewrite time: {performance['query_rewrite_time_seconds']} seconds")
                st.write(f"Answer generation time: {performance['answer_generation_time_seconds']} seconds")
                st.write(f"Total pipeline time: {performance['total_pipeline_time_seconds']} seconds")

with tab_evaluation:
    st.markdown("**Testing & Evaluation**")
    st.caption("These results are designed for manual review during the demo. They are not automatic pass/fail judgments.")

    default_test_cases = get_default_test_cases()
    test_case_options = {test_case["test_name"]: test_case for test_case in default_test_cases}
    st.dataframe(
        [
            {
                "Test Name": test_case["test_name"],
                "Question": test_case["question"],
                "Expected Behavior": test_case["expected_behavior"],
                "Type": test_case["test_type"],
            }
            for test_case in default_test_cases
        ],
        width="stretch",
        hide_index=True,
    )

    if not st.session_state.documents_ready:
        st.warning("Backend index is not ready. Process documents first.")
    else:
        selected_test_name = st.selectbox(
            "Select a predefined test case",
            options=list(test_case_options.keys()),
        )
        run_selected_test_clicked = st.button("Run Selected Test", key="tab_eval_selected")
        run_all_tests_clicked = st.button("Run All Tests", key="tab_eval_all")

        if run_selected_test_clicked:
            try:
                status = get_ollama_status(backend_url, ollama_model_name)
                st.session_state.ollama_status = status
                if not status["ok"]:
                    st.error(status["message"])
                else:
                    with st.spinner("Running selected evaluation test through FastAPI..."):
                        result = _run_api_test_case(
                            backend_url,
                            test_case_options[selected_test_name],
                            ollama_model_name=ollama_model_name,
                            top_k=top_k,
                            max_context_chunks=max_context_chunks,
                            max_context_chars=max_context_chars,
                            answer_temperature=answer_temperature,
                            answer_num_predict=answer_num_predict,
                            use_reranker=use_reranker,
                        )
                    st.session_state.evaluation_results = [result]
                    st.session_state.evaluation_summary = summarize_test_results([result])
            except ApiClientError as exc:
                st.error(str(exc))
                st.session_state.api_status = {"ok": False, "message": str(exc)}

        if run_all_tests_clicked:
            try:
                status = get_ollama_status(backend_url, ollama_model_name)
                st.session_state.ollama_status = status
                if not status["ok"]:
                    st.error(status["message"])
                else:
                    with st.spinner("Running all evaluation tests through FastAPI..."):
                        results = [
                            _run_api_test_case(
                                backend_url,
                                test_case,
                                ollama_model_name=ollama_model_name,
                                top_k=top_k,
                                max_context_chunks=max_context_chunks,
                                max_context_chars=max_context_chars,
                                answer_temperature=answer_temperature,
                                answer_num_predict=answer_num_predict,
                                use_reranker=use_reranker,
                            )
                            for test_case in default_test_cases
                        ]
                    st.session_state.evaluation_results = results
                    st.session_state.evaluation_summary = summarize_test_results(results)
            except ApiClientError as exc:
                st.error(str(exc))
                st.session_state.api_status = {"ok": False, "message": str(exc)}

    if st.session_state.evaluation_summary:
        summary = st.session_state.evaluation_summary
        cols = st.columns(7)
        cols[0].metric("Total tests", summary["total_tests"])
        cols[1].metric("Fallback count", summary["fallback_count"])
        cols[2].metric("Avg latency (s)", summary["average_latency"])
        cols[3].metric("Tests with citations", summary["tests_with_citations"])
        cols[4].metric("Tests with retry", summary["tests_with_retry"])
        cols[5].metric("Avg fact coverage", summary["average_fact_coverage"])
        cols[6].metric("Avg page coverage", summary["average_page_coverage"])

    if st.session_state.evaluation_results:
        for result in st.session_state.evaluation_results:
            with st.expander(result["test_name"], expanded=False):
                st.write(f"Question: {result['question']}")
                st.write(f"Expected behavior: {result['expected_behavior']}")
                st.write(f"Final answer: {result['answer']}")
                st.write(f"Retrieval attempts: {result['retrieval_attempts']}")
                st.write(f"Fallback triggered: {result['fallback_triggered']}")
                st.write(f"Latency: {result['latency_seconds']} seconds")
                st.write(f"Retrieval evaluation: {result['retrieval_evaluation']}")
                st.write(f"Notes: {result['notes']}")

                metrics = result.get("metrics", {})
                if metrics:
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Fact coverage", metrics.get("fact_coverage", 0.0))
                    metric_cols[1].metric("Page coverage", metrics.get("page_coverage", 0.0))
                    metric_cols[2].metric("Retrieval hit", metrics.get("retrieval_hit", 0))
                    metric_cols[3].metric("Fallback correct", metrics.get("fallback_correct", 0))

                    expected_facts = result.get("expected_facts", [])
                    if expected_facts:
                        st.markdown("**Expected facts**")
                        fact_hits = {item.get("label", ""): item.get("matched", False) for item in metrics.get("fact_hits", [])}
                        for fact in expected_facts:
                            label = fact.get("label", "")
                            status = "PASS" if fact_hits.get(label, False) else "MISS"
                            st.write(f"- [{status}] {label}")

                    expected_pages = result.get("expected_pages", [])
                    if expected_pages:
                        st.write(
                            f"Expected pages: {expected_pages} | Matched cited pages: {metrics.get('matched_pages', [])}"
                        )

                if result["citations"]:
                    _render_citations(
                        result["citations"],
                        key_prefix=f"evaluation-{result['test_name']}",
                    )

                agentic_result = result["agentic_result"]
                if agentic_result["retrieved_chunks"]:
                    with st.expander("Retrieved Chunks For This Test", expanded=False):
                        for chunk in agentic_result["retrieved_chunks"]:
                            st.markdown(
                                f"**Rank {chunk['rank']} | Score {chunk['score']:.4f} | {chunk['chunk_id']}**  \n"
                                f"{chunk['source']} | Page {chunk['page_number']}"
                            )
                            st.text_area(
                                label=f"eval-chunk-{result['test_name']}-{chunk['chunk_id']}",
                                value=_truncate_text(chunk["chunk_text"], 900),
                                height=120,
                                disabled=True,
                                label_visibility="collapsed",
                            )
