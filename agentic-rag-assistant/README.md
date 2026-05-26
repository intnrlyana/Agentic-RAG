# Agentic RAG Document Assistant

A document question-answering system built for an AI engineering technical task. The app lets a user upload PDF documents, process them into a retrieval pipeline, and ask grounded questions through a chat interface with page-level citations.

The shipped experience is:

- `app.py` as the main Streamlit frontend
- `api.py` as the FastAPI backend
- `src/` for document processing, retrieval, and answer generation

## What It Does

- Upload one or more PDF documents
- Extract and preprocess page text
- Chunk content for retrieval
- Build embeddings and a FAISS-based index
- Retrieve relevant evidence for each question
- Generate grounded answers with citations
- Show answer details such as retrieved chunks and pipeline metadata

## Tech Stack

- Python
- Streamlit
- FastAPI
- SentenceTransformers
- FAISS
- LlamaIndex-based backend flow
- Groq or Ollama for answer generation

## Project Structure

```text
agentic-rag-assistant/
├── app.py                 # Main Streamlit chat app
├── api.py                 # FastAPI backend
├── pre_app.py             # Earlier prototype version
├── requirements.txt
├── .env.example
├── README.md
├── data/                  # Sample PDFs or local test documents
├── docs/
└── src/
    ├── agentic_rag.py
    ├── api_client.py
    ├── api_service.py
    ├── chunking.py
    ├── citations.py
    ├── document_loader.py
    ├── evidence_selection.py
    ├── llamaindex_backend.py
    ├── preprocessing.py
    ├── query_understanding.py
    ├── utils.py
    └── vector_store.py
```

## System Flow

1. Upload PDFs in the Streamlit app.
2. Streamlit sends the files to the FastAPI backend.
3. The backend extracts text, preprocesses pages, chunks content, and builds retrieval artifacts.
4. When a user asks a question, the backend retrieves relevant chunks and generates a grounded answer.
5. The frontend displays the answer together with citations and supporting details.

## Main Features

- Clean chat-style UI for the final demo
- Multi-PDF upload support
- Per-file enable/disable selection before processing
- Retrieval-backed answers with citations
- Sample starter questions for quick testing
- Backend health check and LLM/provider status check
- Configurable chunking and embedding settings in the processing request

## Requirements

- Python 3.10+
- `pip`
- A configured LLM provider:
  - `Groq` recommended for simpler setup
  - `Ollama` supported for local inference

## Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Recommended Groq configuration:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
RAG_BACKEND=llamaindex
```

Optional Ollama configuration:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2:3b
RAG_BACKEND=llamaindex
```

Notes:

- `app.py` reads `BACKEND_URL` if you want to override the default backend address.
- If `BACKEND_URL` is not set, the frontend uses `http://127.0.0.1:8000`.

## Local Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

Start the FastAPI backend:

```bash
uvicorn api:app --reload
```

The backend will be available at:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

In a second terminal, start the Streamlit app:

```bash
python -m streamlit run app.py
```

The frontend will usually open at:

- `http://localhost:8501`

## How to Use

1. Open the Streamlit app.
2. Upload one or more PDF files in the sidebar.
3. Optionally disable any uploaded file you do not want to include.
4. Click `Process documents`.
5. Ask questions in the chat input.
6. Review the answer, citations, and detail panel.

Example questions:

- `How many public holidays do employees get?`
- `Can I wear jeans to work?`
- `What is the resignation notice period?`
- `Summarize the leave and resignation rules.`

## API Endpoints

Main backend endpoints:

- `GET /health`
- `GET /ollama/status`
- `GET /documents/status`
- `POST /documents/process/local`
- `POST /documents/process/upload`
- `POST /retrieve`
- `POST /ask`

## Notes for Reviewers

This project is structured as a small service-oriented RAG application:

- `app.py` is the user-facing demo layer
- `api.py` exposes document processing and question-answering endpoints
- `src/` contains the retrieval and answer pipeline

The primary submission path is the Streamlit chat app backed by FastAPI.

## Known Limitations

- Retrieval quality depends on PDF text extraction quality.
- Scanned PDFs without extractable text may perform poorly unless OCR is added.
- The backend currently keeps processed artifacts in memory for the running session.
- Large documents or many uploaded PDFs can increase processing and response time.

## Submission Tip

If you are pushing this for interview review, the easiest evaluation package is:

- this GitHub repository
- a short demo video or screenshots
- an optional hosted version if it is stable
