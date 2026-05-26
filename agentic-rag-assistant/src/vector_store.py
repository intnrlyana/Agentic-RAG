"""Vector store helpers for hybrid retrieval, FAISS search, and reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


ChunkRecord = dict[str, Any]
SearchResult = dict[str, Any]
LexicalIndex = dict[str, Any]

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]{3,}\b")
_STOPWORDS = {
    "about",
    "after",
    "before",
    "been",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "over",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def load_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Load a CrossEncoder reranker model."""
    return CrossEncoder(model_name)


def create_faiss_index(
    chunks: list[ChunkRecord],
    model: SentenceTransformer,
) -> tuple[faiss.IndexFlatIP | None, list[ChunkRecord], np.ndarray]:
    """Create a FAISS index from chunk text and keep metadata aligned with vectors."""
    metadata = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_text") and not chunk.get("is_low_information", False)
    ]
    if not metadata:
        return None, [], np.array([], dtype=np.float32)

    texts = [chunk["chunk_text"] for chunk in metadata]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata, embeddings


def create_lexical_index(chunks: list[ChunkRecord]) -> tuple[list[ChunkRecord], LexicalIndex]:
    """Build a lightweight BM25-style lexical index aligned with chunk metadata."""
    metadata = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_text") and not chunk.get("is_low_information", False)
    ]
    if not metadata:
        return [], {
            "doc_term_freqs": [],
            "doc_lengths": [],
            "term_doc_counts": {},
            "avg_doc_length": 0.0,
        }

    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    term_doc_counts: Counter[str] = Counter()

    for chunk in metadata:
        tokens = _tokenize_terms_with_frequency(chunk["chunk_text"])
        term_freqs = Counter(tokens)
        doc_term_freqs.append(term_freqs)
        doc_lengths.append(sum(term_freqs.values()))
        term_doc_counts.update(term_freqs.keys())

    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    lexical_index: LexicalIndex = {
        "doc_term_freqs": doc_term_freqs,
        "doc_lengths": doc_lengths,
        "term_doc_counts": dict(term_doc_counts),
        "avg_doc_length": avg_doc_length,
    }
    return metadata, lexical_index


def search_similar_chunks(
    query: str,
    index: faiss.IndexFlatIP | None,
    metadata: list[ChunkRecord],
    model: SentenceTransformer,
    lexical_index: LexicalIndex | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """Run hybrid retrieval using dense search, lexical search, and overlap fusion."""
    if not query.strip() or not metadata:
        return []

    search_k = min(max(top_k * 3, top_k), len(metadata))
    semantic_scores_by_idx: dict[int, float] = {}
    semantic_candidate_ids: set[int] = set()

    if index is not None:
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, search_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            semantic_scores_by_idx[int(idx)] = float(score)
            semantic_candidate_ids.add(int(idx))

    lexical_scores_by_idx = _score_lexical_matches(query, lexical_index, len(metadata))
    lexical_ranked_ids = sorted(
        lexical_scores_by_idx,
        key=lambda idx: lexical_scores_by_idx[idx],
        reverse=True,
    )[:search_k]

    candidate_ids = semantic_candidate_ids | set(lexical_ranked_ids)
    if not candidate_ids:
        return []

    semantic_normalized = _min_max_normalize(semantic_scores_by_idx, candidate_ids)
    lexical_normalized = _min_max_normalize(lexical_scores_by_idx, candidate_ids)

    candidates: list[SearchResult] = []
    for idx in candidate_ids:
        chunk = metadata[idx]
        semantic_score = semantic_scores_by_idx.get(idx, 0.0)
        lexical_score = lexical_scores_by_idx.get(idx, 0.0)
        keyword_overlap = calculate_keyword_overlap(query, chunk["chunk_text"])
        heading_boost = calculate_heading_boost(query, chunk)
        combined_score = (
            0.55 * semantic_normalized.get(idx, 0.0)
            + 0.25 * lexical_normalized.get(idx, 0.0)
            + 0.10 * keyword_overlap
            + 0.10 * heading_boost
        )
        candidates.append(
            {
                "faiss_score": semantic_score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "keyword_overlap": keyword_overlap,
                "heading_boost": heading_boost,
                "score": combined_score,
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "source_type": chunk["source_type"],
                "original_text_length": chunk.get("original_text_length"),
                "cleaned_text_length": chunk.get("cleaned_text_length"),
                "section_title": chunk.get("section_title"),
                "subsection_title": chunk.get("subsection_title"),
                "heading_path": chunk.get("heading_path"),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["score"],
            item.get("semantic_score", 0.0),
            item.get("lexical_score", 0.0),
        ),
        reverse=True,
    )
    return _assign_ranks(candidates[:top_k])


def rerank_chunks(
    query: str,
    retrieved_chunks: list[SearchResult],
    reranker: CrossEncoder | None,
    top_n: int,
) -> list[SearchResult]:
    """Rerank retrieved chunks with a CrossEncoder model."""
    if not query.strip() or not retrieved_chunks:
        return []
    if reranker is None:
        return _assign_ranks(retrieved_chunks[:top_n])

    pairs = [(query, chunk["chunk_text"]) for chunk in retrieved_chunks]
    reranker_scores = reranker.predict(pairs, show_progress_bar=False)
    reranked_chunks: list[SearchResult] = []
    for chunk, reranker_score in zip(retrieved_chunks, reranker_scores):
        reranked_chunk = dict(chunk)
        reranked_chunk["reranker_score"] = float(reranker_score)
        reranked_chunks.append(reranked_chunk)

    reranked_chunks.sort(
        key=lambda item: (
            item.get("reranker_score", float("-inf")),
            item.get("score", 0.0),
        ),
        reverse=True,
    )
    return _assign_ranks(reranked_chunks[:top_n])


def summarize_vector_store(
    chunks: list[ChunkRecord],
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Summarize vector-store-ready chunks for UI display."""
    chunks_by_document: dict[str, int] = {}
    indexed_chunks = [chunk for chunk in chunks if not chunk.get("is_low_information", False)]
    for chunk in indexed_chunks:
        source = chunk["source"]
        chunks_by_document[source] = chunks_by_document.get(source, 0) + 1

    return {
        "total_vectors": len(indexed_chunks),
        "total_documents": len(chunks_by_document),
        "embedding_model": model_name,
        "chunks_by_document": chunks_by_document,
        "filtered_low_information_chunks": len(chunks) - len(indexed_chunks),
    }


def calculate_keyword_overlap(query: str, text: str) -> float:
    """Compute a lightweight keyword overlap ratio for retrieval evaluation."""
    query_terms = _tokenize_query_terms(query)
    if not query_terms:
        return 0.0
    text_terms = _tokenize_query_terms(text)
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def calculate_heading_boost(query: str, chunk: ChunkRecord) -> float:
    """Boost chunks whose heading metadata aligns with the query."""
    query_terms = _tokenize_query_terms(query)
    if not query_terms:
        return 0.0

    heading_terms = _tokenize_query_terms(
        " ".join(
            value
            for value in [
                chunk.get("section_title", ""),
                chunk.get("subsection_title", ""),
                chunk.get("heading_path", ""),
            ]
            if value
        )
    )
    if not heading_terms:
        return 0.0

    overlap_ratio = len(query_terms & heading_terms) / len(query_terms)
    if overlap_ratio >= 0.5:
        return min(1.0, overlap_ratio + 0.2)
    return overlap_ratio


def _score_lexical_matches(
    query: str,
    lexical_index: LexicalIndex | None,
    document_count: int,
) -> dict[int, float]:
    """Compute BM25-style lexical scores for each indexed chunk."""
    if not lexical_index or document_count <= 0:
        return {}

    query_terms = _tokenize_terms_with_frequency(query)
    if not query_terms:
        return {}

    doc_term_freqs: list[Counter[str]] = lexical_index.get("doc_term_freqs", [])
    doc_lengths: list[int] = lexical_index.get("doc_lengths", [])
    term_doc_counts: dict[str, int] = lexical_index.get("term_doc_counts", {})
    avg_doc_length = float(lexical_index.get("avg_doc_length", 0.0) or 0.0)
    if not doc_term_freqs or not doc_lengths or avg_doc_length <= 0.0:
        return {}

    k1 = 1.5
    b = 0.75
    scores_by_idx: dict[int, float] = {}
    unique_query_terms = list(dict.fromkeys(query_terms))

    for idx, term_freqs in enumerate(doc_term_freqs):
        doc_length = doc_lengths[idx] if idx < len(doc_lengths) else 0
        if doc_length <= 0:
            continue

        score = 0.0
        for term in unique_query_terms:
            term_frequency = term_freqs.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = term_doc_counts.get(term, 0)
            idf = math.log(1.0 + ((document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            numerator = term_frequency * (k1 + 1.0)
            denominator = term_frequency + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)

        if score > 0.0:
            scores_by_idx[idx] = score

    return scores_by_idx


def _assign_ranks(chunks: list[SearchResult]) -> list[SearchResult]:
    """Assign display ranks to a chunk list."""
    ranked_chunks: list[SearchResult] = []
    for rank, chunk in enumerate(chunks, start=1):
        ranked_chunk = dict(chunk)
        ranked_chunk["rank"] = rank
        ranked_chunks.append(ranked_chunk)
    return ranked_chunks


def _min_max_normalize(
    scores_by_idx: dict[int, float],
    candidate_ids: set[int],
) -> dict[int, float]:
    """Normalize candidate scores into a stable 0..1 range."""
    filtered_scores = {
        idx: float(scores_by_idx.get(idx, 0.0))
        for idx in candidate_ids
    }
    if not filtered_scores:
        return {}

    minimum = min(filtered_scores.values())
    maximum = max(filtered_scores.values())
    if math.isclose(minimum, maximum):
        return {
            idx: (1.0 if score > 0.0 else 0.0)
            for idx, score in filtered_scores.items()
        }

    scale = maximum - minimum
    return {
        idx: (score - minimum) / scale
        for idx, score in filtered_scores.items()
    }


def _tokenize_query_terms(text: str) -> set[str]:
    """Tokenize and lightly normalize text for generic keyword overlap."""
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    }


def _tokenize_terms_with_frequency(text: str) -> list[str]:
    """Tokenize text into normalized terms while preserving repeated matches."""
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    ]
