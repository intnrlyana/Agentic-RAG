# Agentic RAG Flow

## Current Shipped Flow

The final system is built as:

- `app.py` as the main Streamlit frontend
- `api.py` as the FastAPI backend
- `src/` as the shared retrieval and answering pipeline

The user-facing flow is:

1. The user uploads one or more `PDF` or `DOCX` files in Streamlit.
2. Streamlit sends the files to the FastAPI backend.
3. The backend extracts text, preprocesses it, chunks it, and builds retrieval artifacts.
4. The user asks a question in the chat UI.
5. The backend retrieves relevant evidence and generates a grounded answer.
6. The frontend shows the answer, citations, and supporting details.

## Workflow Diagram

```text
Upload PDF/DOCX
  |
  v
FastAPI Processing
  |
  +--> Text Extraction
  |
  +--> Preprocessing
  |
  +--> Chunking
  |
  +--> Embedding + Retrieval Index Build
  |
  v
Question From User
  |
  v
Query Planning / Rewriting
  |
  v
LlamaIndex Retrieval
  |
  v
Reranking + Evidence Selection
  |
  +--> Evidence strong enough? ---- yes ----> Grounded Answer Generation
  |                                              |
  |                                              v
  |                                        Citations Attached
  |                                              |
  |                                              v
  |                                         Final Answer
  |
  +--> no ----> Retry / fallback path
```

## Main Pipeline Stages

### 1. Document Ingestion

- Uploaded documents are accepted through the Streamlit UI.
- The current final system supports:
  - `.pdf`
  - `.docx`
- For scanned PDF pages with little or no embedded text, the loader can fall back to OCR.
- Extracted content is normalized into page-like records so the downstream pipeline can treat sources consistently.

### 2. Preprocessing

- Extracted text is lightly cleaned.
- The goal is to preserve meaning while reducing noise from raw document extraction.
- Empty or very low-value content is tracked so the processing summary remains visible.

### 3. Chunking

- Processed content is split into retrieval-friendly chunks.
- Metadata such as source name, page number, and chunk identifiers are preserved.
- This metadata is later used for citations and answer inspection.

### 4. Retrieval

- The current backend path is LlamaIndex-based.
- Retrieved chunks are enriched with overlap signals and reranked before answer generation.
- The system still keeps explicit chunk metadata and retrieval details for explainability.

### 5. Agentic Answering

- The system can take a lighter path for simple questions or a more structured path for broader multi-part questions.
- It evaluates whether retrieval quality looks strong enough.
- If retrieval is weak, the system can retry through a broader or adjusted path before falling back.

### 6. Grounding and Citations

- Answers are generated from retrieved evidence rather than unrestricted free-form generation.
- Citations are attached using source and page metadata from the retrieved chunks.
- The UI also exposes retrieved chunk snippets and performance details for inspection.

## Example Successful Case

Question:

```text
What is the resignation notice period?
```

Typical behavior:

- retrieval finds the relevant resignation-policy chunks
- the answer is generated from those chunks
- citations point back to the supporting document pages

## Example Fallback Case

Question:

```text
What is the moon allowance policy?
```

Typical behavior:

- retrieval remains weak or irrelevant
- the pipeline avoids overcommitting
- the system returns a fallback-style grounded response instead of inventing policy content

## Why This Flow Fits The Project

- It keeps the system architecture easy to follow.
- It separates the frontend from the backend cleanly.
- It preserves retrieval evidence, citations, and inspection details.
- It stays grounded in the uploaded documents instead of acting like a generic chatbot.
