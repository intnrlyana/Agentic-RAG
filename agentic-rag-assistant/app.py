from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.document_loader import (
    extract_text_from_pdf_file,
    load_pdfs_from_data_folder,
    summarize_extraction,
)
from src.preprocessing import preprocess_pages, summarize_preprocessing


DATA_DIR = Path("data")


st.set_page_config(
    page_title="Agentic RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("Agentic RAG Document Assistant")
st.caption("Phase 3: extract PDF text, then lightly preprocess it for downstream RAG steps.")

with st.sidebar:
    st.header("Document Source")
    source_mode = st.radio(
        "Choose how to load PDFs",
        options=["Upload PDFs", "Load from data/ folder"],
    )

    uploaded_files = []
    selected_source_names: list[str] = []

    if source_mode == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Add one or more PDF documents",
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

extract_clicked = st.button("Extract PDF Text", type="primary")

pages: list[dict] = []
attempted_pdf_count = len(selected_source_names)

if extract_clicked:
    if source_mode == "Upload PDFs":
        if not uploaded_files:
            st.warning("Select at least one PDF file to extract.")
        else:
            for uploaded_file in uploaded_files:
                pages.extend(extract_text_from_pdf_file(uploaded_file, source_type="uploaded"))
    else:
        pages = load_pdfs_from_data_folder(data_dir=str(DATA_DIR))
        if not selected_source_names:
            st.warning("Add PDF files to the `data/` folder and run extraction again.")

st.subheader("Workspace")
st.write(
    "This phase focuses on PDF loading and light text preprocessing only. "
    "Chunking, embeddings, vector storage, and answer generation are intentionally deferred."
)

if extract_clicked and pages:
    summary = summarize_extraction(pages)
    processed_pages = preprocess_pages(pages)
    preprocessing_summary = summarize_preprocessing(processed_pages)
    retained_pages = [page for page in processed_pages if not page["was_skipped"]]
    skipped_pages = [page for page in processed_pages if page["was_skipped"]]
    extracted_source_names = summary["source_filenames"]
    failed_or_empty_sources = sorted(set(selected_source_names) - set(extracted_source_names))

    st.subheader("Extraction Summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric("PDFs selected", attempted_pdf_count)
    metric_columns[1].metric("PDFs with pages extracted", summary["pdf_count"])
    metric_columns[2].metric("Pages extracted", summary["total_pages_extracted"])
    metric_columns[3].metric("Empty pages", summary["empty_or_skipped_pages"])

    st.write("Source files:", ", ".join(selected_source_names))
    if failed_or_empty_sources:
        st.warning(
            "These PDFs returned no readable pages and may be empty, image-only, or invalid: "
            + ", ".join(failed_or_empty_sources)
        )

    st.subheader("Preprocessing Summary")
    preprocessing_metrics = st.columns(5)
    preprocessing_metrics[0].metric("Total pages", preprocessing_summary["total_pages"])
    preprocessing_metrics[1].metric("Retained pages", preprocessing_summary["retained_pages"])
    preprocessing_metrics[2].metric("Skipped pages", preprocessing_summary["skipped_pages"])
    preprocessing_metrics[3].metric(
        "Original characters",
        preprocessing_summary["total_original_characters"],
    )
    preprocessing_metrics[4].metric(
        "Cleaned characters",
        preprocessing_summary["total_cleaned_characters"],
    )

    if preprocessing_summary["skipped_by_reason"]:
        skipped_reason_text = ", ".join(
            f"{reason}: {count}" for reason, count in preprocessing_summary["skipped_by_reason"].items()
        )
        st.write("Skipped by reason:", skipped_reason_text)

    st.subheader("Original vs Cleaned Preview")
    preview_blocks = []
    preview_limit = min(5, len(processed_pages))
    for raw_page, processed_page in zip(pages[:preview_limit], processed_pages[:preview_limit]):
        original_preview = raw_page["text"] or "[No extractable text found on this page]"
        cleaned_preview = processed_page["text"] or "[No retained text after preprocessing]"
        preview_blocks.append(
            f"{processed_page['source']} | Page {processed_page['page_number']} | {processed_page['source_type']}\n"
            f"Original:\n{original_preview[:500]}\n\n"
            f"Cleaned:\n{cleaned_preview[:500]}"
        )
    st.text_area(
        "Preview of the first processed pages",
        value="\n\n---\n\n".join(preview_blocks),
        height=420,
        disabled=True,
    )

    st.subheader("Retained Pages")
    pages_by_source: dict[str, list[dict]] = {}
    for page in retained_pages:
        pages_by_source.setdefault(page["source"], []).append(page)

    for source_name, source_pages in pages_by_source.items():
        source_type = source_pages[0]["source_type"]
        with st.expander(f"{source_name} ({len(source_pages)} pages, {source_type})", expanded=False):
            for page in source_pages:
                page_text = page["text"] or "[No retained text after preprocessing]"
                st.markdown(f"**Page {page['page_number']}**")
                st.caption(
                    f"Original length: {page['original_text_length']} chars | "
                    f"Cleaned length: {page['cleaned_text_length']} chars"
                )
                st.text_area(
                    label=f"{source_name}-page-{page['page_number']}",
                    value=page_text,
                    height=180,
                    disabled=True,
                    label_visibility="collapsed",
                )

    if skipped_pages:
        with st.expander(f"Skipped Pages ({len(skipped_pages)})", expanded=False):
            for page in skipped_pages:
                st.markdown(f"**{page['source']} | Page {page['page_number']}**")
                st.caption(
                    f"Reason: {page['skip_reason']} | "
                    f"Original length: {page['original_text_length']} chars | "
                    f"Cleaned length: {page['cleaned_text_length']} chars"
                )
                st.text_area(
                    label=f"skipped-{page['source']}-{page['page_number']}",
                    value=page["text"] or "[No retained text after preprocessing]",
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                )
elif extract_clicked:
    st.warning("No pages were extracted from the selected PDFs.")

user_message = st.chat_input("Ask a question about your uploaded documents...")
if user_message:
    st.chat_message("user").write(user_message)
    st.chat_message("assistant").write(
        "Chat-based retrieval is still a placeholder. Phase 3 only covers extraction and preprocessing."
    )
