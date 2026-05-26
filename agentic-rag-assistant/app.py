"""Main Streamlit app for the final document assistant."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.api_client import (
    ApiClientError,
    ask_agentic_rag,
    get_api_health,
    get_ollama_status,
    process_uploaded_documents,
)


_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
if not (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env.example", override=False)


st.set_page_config(
    page_title="Agentic RAG Document Fact-Based",
    page_icon="A",
    layout="centered",
    initial_sidebar_state="collapsed",
)


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL_NAME",
    os.getenv("LLM_MODEL_NAME", os.getenv("OLLAMA_MODEL_NAME", "llama-3.3-70b-versatile")),
)

SAMPLE_QUESTIONS = [
    "How many public holidays do employees get?",
    "Can I wear jeans to work?",
    "What is the resignation notice period?",
    "Summarize the leave and resignation rules.",
]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f3ea;
            --ink: #1f2b24;
            --muted: #6b746c;
            --line: #dfd5c2;
            --panel: rgba(255, 252, 246, 0.92);
            --panel-2: rgba(247, 241, 229, 0.92);
            --hero: #173f35;
            --hero-2: #245548;
            --accent: #c9842d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201,132,45,0.10), transparent 22%),
                radial-gradient(circle at top right, rgba(23,63,53,0.10), transparent 18%),
                linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 860px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        .shell {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }

        .hero {
            background: linear-gradient(135deg, var(--hero) 0%, var(--hero-2) 100%);
            border-radius: 24px;
            padding: 1.25rem 1.3rem;
            color: #f8f5ee;
            box-shadow: 0 16px 34px rgba(24, 37, 31, 0.15);
        }

        .hero h1 {
            margin: 0;
            font-size: 1.85rem;
            letter-spacing: -0.03em;
            color: #f8f5ee;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            color: rgba(248,245,238,0.84);
            max-width: 640px;
        }

        .hero-pills {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }

        .hero-pill {
            border-radius: 999px;
            padding: 0.38rem 0.66rem;
            font-size: 0.79rem;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.16);
        }

        .panel {
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 22px rgba(51, 45, 34, 0.06);
        }

        .panel h3 {
            margin: 0 0 0.2rem 0;
            font-size: 1rem;
            letter-spacing: -0.02em;
        }

        div[data-testid="stVerticalBlock"]:has(#sample-panel-anchor) {
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 22px rgba(51, 45, 34, 0.06);
            margin-bottom: 0.9rem;
        }

        .panel p {
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .source-chip {
            display: inline-block;
            margin: 0.3rem 0.35rem 0 0;
            padding: 0.34rem 0.58rem;
            border-radius: 999px;
            background: #efe2c5;
            border: 1px solid #ddc59b;
            color: #4e3f28;
            font-size: 0.8rem;
        }

        .tiny-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 20px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 251, 244, 0.75);
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid #174739;
            background: linear-gradient(180deg, #1b6a53 0%, #164f40 100%);
            color: white;
            font-weight: 600;
            padding: 0.52rem 0.95rem;
        }

        .stDownloadButton > button,
        .stFileUploader button {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "documents_ready": False,
        "processing_response": None,
        "chat_history": [],
        "api_status": None,
        "llm_status": None,
        "sample_prompt": "",
        "enabled_uploads": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _refresh_status(backend_url: str, model_name: str) -> None:
    try:
        st.session_state.api_status = get_api_health(backend_url)
    except ApiClientError as exc:
        st.session_state.api_status = {"ok": False, "message": str(exc)}

    try:
        st.session_state.llm_status = get_ollama_status(backend_url, model_name)
    except ApiClientError as exc:
        st.session_state.llm_status = {"ok": False, "message": str(exc)}


def _render_header(backend_url: str, model_name: str) -> None:
    if st.session_state.api_status is None or st.session_state.llm_status is None:
        _refresh_status(backend_url, model_name)

    st.markdown(
        """
        <div class="hero">
            <h1>Agentic RAG Document Fact-Based</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sources(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    rendered = []
    seen: set[tuple[str, Any]] = set()
    for citation in citations:
        key = (str(citation.get("source", "")), citation.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        rendered.append(
            f"<span class='source-chip'>{html.escape(str(citation.get('source', 'Source')))} • Page {citation.get('page_number', '?')}</span>"
        )
    st.markdown("".join(rendered), unsafe_allow_html=True)


def _truncate(text: str, limit: int = 900) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _file_identity(uploaded_file: Any) -> str:
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def _append_chat(question: str, result: dict[str, Any]) -> None:
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result.get("final_answer", ""),
            "citations": result.get("citations", []),
            "details": result,
        }
    )


def _process_documents(
    backend_url: str,
    uploaded_files: list[Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    model_name: str,
) -> None:
    if not uploaded_files:
        st.warning("Enable at least one document before processing.")
        return

    try:
        with st.spinner("Processing documents..."):
            response = process_uploaded_documents(
                backend_url,
                uploaded_files,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model_name=embedding_model_name,
            )
        st.session_state.documents_ready = True
        st.session_state.processing_response = response
        st.session_state.chat_history = []
        _refresh_status(backend_url, model_name)
        st.success("Documents processed. You can start chatting now.")
    except ApiClientError as exc:
        st.error(str(exc))


def _ask_question(
    backend_url: str,
    question: str,
    *,
    model_name: str,
    top_k: int,
    max_context_chunks: int,
    max_context_chars: int,
    answer_temperature: float,
    answer_num_predict: int,
) -> bool:
    if not st.session_state.documents_ready:
        st.warning("Process documents first.")
        return False
    if not question.strip():
        return False

    try:
        with st.spinner("Thinking..."):
            st.session_state.llm_status = get_ollama_status(backend_url, model_name)
            if not st.session_state.llm_status.get("ok"):
                st.error(st.session_state.llm_status.get("message", "Model unavailable."))
                return False
            result = ask_agentic_rag(
                backend_url,
                question=question,
                llm_model_name=model_name,
                top_k=top_k,
                max_context_chunks=max_context_chunks,
                max_chars_per_chunk=max_context_chars,
                answer_temperature=answer_temperature,
                answer_num_predict=answer_num_predict,
                use_reranker=True,
            )
        _append_chat(question, result)
        return True
    except ApiClientError as exc:
        st.error(str(exc))
        return False


def _render_upload_panel(
    backend_url: str,
    model_name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> None:
    process = st.session_state.processing_response or {}
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="sidebar_uploaded_files",
    )
    enabled_uploads = st.session_state.enabled_uploads
    uploaded_files = uploaded_files or []
    current_ids = {_file_identity(uploaded_file) for uploaded_file in uploaded_files}

    for stale_id in list(enabled_uploads):
        if stale_id not in current_ids:
            enabled_uploads.pop(stale_id, None)

    for uploaded_file in uploaded_files:
        enabled_uploads.setdefault(_file_identity(uploaded_file), True)

    if uploaded_files:
        st.markdown("**Selected documents**")
        for uploaded_file in uploaded_files:
            file_id = _file_identity(uploaded_file)
            enabled_uploads[file_id] = st.checkbox(
                uploaded_file.name,
                value=bool(enabled_uploads.get(file_id, True)),
                key=f"enabled-upload-{file_id}",
            )

    selected_files = [
        uploaded_file
        for uploaded_file in uploaded_files
        if enabled_uploads.get(_file_identity(uploaded_file), True)
    ]

    if st.button("Process documents", use_container_width=True):
        _process_documents(
            backend_url,
            selected_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name,
            model_name=model_name,
        )
    if st.button("Refresh status", use_container_width=True):
        _refresh_status(backend_url, model_name)

    if uploaded_files:
        st.caption(f"Uploaded: {len(uploaded_files)} file(s) • Enabled: {len(selected_files)}")


def _render_samples() -> None:
    with st.container():
        st.markdown("<div id='sample-panel-anchor'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel" style="background: transparent; border: 0; box-shadow: none; padding: 0; margin-bottom: 0.85rem;">
                <h3>Try one of these</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(2, gap="small")
        for idx, sample in enumerate(SAMPLE_QUESTIONS):
            with cols[idx % 2]:
                if st.button(sample, key=f"sample-{idx}", use_container_width=True):
                    st.session_state.sample_prompt = sample


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            if st.session_state.documents_ready:
                st.markdown("Document ready")
            else:
                st.markdown("Upload document first to get started")
        return

    for idx, item in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            _render_sources(item.get("citations", []))
            details = item.get("details", {})
            with st.expander("Details", expanded=False):
                if details.get("primary_evidence"):
                    st.markdown("**Primary evidence**")
                    st.write(details["primary_evidence"])
                if details.get("retrieved_chunks"):
                    st.markdown("**Top retrieved chunks**")
                    for chunk in details["retrieved_chunks"][:3]:
                        st.caption(f"{chunk.get('source', 'Source')} · Page {chunk.get('page_number', '?')}")
                        st.write(_truncate(chunk.get("chunk_text", "")))
                if details.get("performance"):
                    st.markdown("**Performance**")
                    st.json(details["performance"], expanded=False)


_inject_styles()
_init_state()

backend_url = DEFAULT_BACKEND_URL
model_name = DEFAULT_MODEL
top_k = 8
max_context_chunks = 3
max_context_chars = 1000
answer_temperature = 0.15
answer_num_predict = 220
chunk_size = 900
chunk_overlap = 180
embedding_model_name = "BAAI/bge-small-en-v1.5"

st.markdown("<div class='shell'>", unsafe_allow_html=True)
_render_header(backend_url, model_name)

with st.sidebar:
    st.markdown("### Documents")
    _render_upload_panel(
        backend_url,
        model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )

_render_samples()
_render_chat_history()

prefill_question = st.session_state.pop("sample_prompt", "")
question = st.chat_input(
    "Ask about the uploaded documents...",
    disabled=not st.session_state.documents_ready,
)
if prefill_question and st.session_state.documents_ready:
    submitted = _ask_question(
        backend_url,
        prefill_question,
        model_name=model_name,
        top_k=top_k,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
        answer_temperature=answer_temperature,
        answer_num_predict=answer_num_predict,
    )
    if submitted:
        st.rerun()

if question:
    submitted = _ask_question(
        backend_url,
        question,
        model_name=model_name,
        top_k=top_k,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
        answer_temperature=answer_temperature,
        answer_num_predict=answer_num_predict,
    )
    if submitted:
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
