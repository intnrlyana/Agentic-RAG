# Agentic RAG Document Assistant

Phase 3 scaffold for an interview demo project built with Python and Streamlit.

## Goal

This phase sets up:

- A clean Python project structure
- A Streamlit interface for PDF extraction
- Local `data/` folder PDF support
- Uploaded PDF support
- Page-level extraction metadata
- Light, meaning-preserving preprocessing for extracted pages
- Environment variable and dependency scaffolding

This phase does not implement chunking, embeddings, FAISS indexing, retrieval, or answer generation yet.

## Project Structure

```text
agentic-rag-assistant/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── agentic_rag.py
│   └── utils.py
└── data/
```

## Setup

### 1. Open in VS Code

Open the `agentic-rag-assistant` folder as the workspace root.

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add API keys when needed.

macOS/Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## Phase 3 Usage

### Option 1: Load PDFs from the local `data/` folder

Place any `.pdf` files inside:

```text
agentic-rag-assistant/data/
```

Example:

```text
data/
├── Employee-Handbook.pdf
├── Employee-Handbook-Sample-Malaysia-HR-Forum-v3.pdf
└── sample_policy_and_procedures_manual.pdf
```

Then run:

```bash
streamlit run app.py
```

In the app sidebar:

- Choose `Load from data/ folder`
- Click `Extract PDF Text`

### Option 2: Upload PDFs manually in Streamlit

In the app sidebar:

- Choose `Upload PDFs`
- Select one or more PDF files
- Click `Extract PDF Text`

## Why Preprocessing Matters for RAG

PDF extraction usually produces noisy text:

- irregular spacing
- repeated blank lines
- isolated page-number artifacts
- pages with too little text to be useful

For RAG, preprocessing should be light. We want cleaner retrieval input without changing the meaning of the source. That is why this project:

- keeps punctuation
- keeps casing
- keeps headings and sentence structure
- does not remove stopwords
- does not apply stemming or lemmatization

This preserves factual wording so later retrieval and answer generation remain faithful to the documents.

## What Phase 3 Displays

After extraction and preprocessing, the app shows:

- Number of PDFs selected
- Number of PDFs with extracted pages
- Total pages extracted
- Number of empty pages
- Source filename list
- Preprocessing summary
- Original vs cleaned text preview
- Expandable retained page output
- Expandable skipped page output

Each extracted page is normalized into a dictionary like this:

```python
{
    "source": "Employee-Handbook.pdf",
    "page_number": 1,
    "text": "extracted page text here",
    "source_type": "local"
}
```

Each processed page keeps the original metadata and adds preprocessing metadata:

```python
{
    "source": "Employee-Handbook.pdf",
    "page_number": 1,
    "source_type": "local",
    "text": "cleaned page text here",
    "original_text_length": 1540,
    "cleaned_text_length": 1472,
    "was_skipped": False,
    "skip_reason": None
}
```

## Backend Module

The PDF loading logic is kept modular in `src/document_loader.py` with these functions:

- `extract_text_from_pdf_file(file, source_type="uploaded")`
- `extract_text_from_pdf_path(pdf_path, source_type="local")`
- `load_pdfs_from_data_folder(data_dir="data")`
- `summarize_extraction(pages)`

The loader is designed to:

- Extract text page by page
- Keep empty pages as records with empty text
- Skip bad PDFs without crashing the app
- Support uploaded and local document sources

The preprocessing logic is isolated in `src/preprocessing.py` with these functions:

- `clean_text(text)`
- `preprocess_pages(pages, min_text_length=30)`
- `summarize_preprocessing(processed_pages)`

The preprocessing layer is designed to:

- normalize whitespace conservatively
- remove repeated blank lines
- drop obvious standalone page-number artifacts where safe
- keep factual wording intact
- mark empty or too-short pages as skipped instead of silently deleting them

## Testing Local PDF Extraction and Preprocessing

1. Put one or more `.pdf` files in `data/`
2. Start the app with `python -m streamlit run app.py`
3. Select `Load from data/ folder`
4. Click `Extract PDF Text`
5. Review the extraction summary, preprocessing summary, retained pages, and skipped pages in the browser

## Next Phases

- Text chunking
- Embeddings with `sentence-transformers`
- FAISS vector store integration
- Agentic orchestration for retrieval and answer synthesis
- OpenAI or Gemini API integration
