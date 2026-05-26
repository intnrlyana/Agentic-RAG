"""Local-only LlamaIndex backend helpers for incremental migration."""

from __future__ import annotations

from typing import Any


def llamaindex_is_available() -> bool:
    """Return whether the required LlamaIndex packages are importable."""
    try:
        from llama_index.core import Document, VectorStoreIndex  # noqa: F401
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: F401
        return True
    except Exception:
        return False


def build_llamaindex_documents(chunks: list[dict[str, Any]]) -> list[Any]:
    """Convert chunk records into LlamaIndex Document objects."""
    from llama_index.core import Document

    documents: list[Any] = []
    for chunk in chunks:
        text = (chunk.get("chunk_text") or "").strip()
        if not text:
            continue
        metadata = {
            "source": chunk.get("source"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("chunk_index"),
            "heading": chunk.get("heading"),
        }
        documents.append(
            Document(
                text=text,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )
    return documents


def build_llamaindex_index(
    chunks: list[dict[str, Any]],
    *,
    embedding_model_name: str,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build a local LlamaIndex vector index from existing chunk records."""
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    documents = build_llamaindex_documents(chunks)
    if not documents:
        return None, [], {
            "available": True,
            "built": False,
            "reason": "No non-empty chunks were available for LlamaIndex.",
            "document_count": 0,
            "embedding_model": embedding_model_name,
        }

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    summary = {
        "available": True,
        "built": True,
        "reason": "LlamaIndex vector index built successfully.",
        "document_count": len(documents),
        "embedding_model": embedding_model_name,
    }
    return index, documents, summary


def retrieve_llamaindex_chunks(
    query: str,
    *,
    index: Any,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve top chunks through LlamaIndex and map them back to existing chunk records."""
    if index is None or not query.strip():
        return []

    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    chunk_lookup = {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in chunks
        if chunk.get("chunk_id")
    }

    mapped_chunks: list[dict[str, Any]] = []
    for rank, node_with_score in enumerate(retrieved_nodes, start=1):
        node = getattr(node_with_score, "node", None)
        score = float(getattr(node_with_score, "score", 0.0) or 0.0)
        metadata = getattr(node, "metadata", {}) or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        source_chunk = chunk_lookup.get(chunk_id)
        if source_chunk is None:
            text = ""
            if node is not None and hasattr(node, "get_content"):
                text = node.get_content()
            source_chunk = {
                "source": metadata.get("source", "unknown"),
                "page_number": int(metadata.get("page_number", 0) or 0),
                "chunk_id": chunk_id or f"llamaindex_rank_{rank}",
                "chunk_text": text,
            }

        mapped_chunks.append(
            {
                **source_chunk,
                "rank": rank,
                "score": score,
                "faiss_score": score,
                "semantic_score": score,
                "keyword_overlap": 0.0,
                "reranker_score": None,
                "matched_queries": [query],
            }
        )

    return mapped_chunks
