"""
retriever.py — Retriever for the RAG Document Q&A System.

Wraps the persisted FAISS index with:
- a LangChain ``VectorStoreRetriever`` (used by the QA chain), and
- a ``similarity_search`` helper that supports optional metadata filtering
  by company, quarter, and/or year.
"""
import sys
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TOP_K, VECTORSTORE_DIR


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def get_retriever(
    persist_dir: Path = VECTORSTORE_DIR,
    k: int = TOP_K,
):
    """
    Build a LangChain ``VectorStoreRetriever`` from the persisted FAISS index.

    Args:
        persist_dir: Path to the saved FAISS index directory.
        k: Number of chunks to return per query.

    Returns:
        A :class:`~langchain.vectorstores.base.VectorStoreRetriever` instance
        ready to plug into a ``RetrievalQA`` chain.
    """
    from embedder import load_vectorstore

    vectorstore = load_vectorstore(persist_dir)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Similarity search with metadata filtering
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = TOP_K,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[str] = None,
    persist_dir: Path = VECTORSTORE_DIR,
) -> List[Document]:
    """
    Retrieve the most relevant chunks for *query* with optional filters.

    Filtering is applied as a post-processing step on a larger candidate set
    (``fetch_k = k × 4``) to ensure enough results remain after pruning.

    Args:
        query:       Natural language question.
        k:           Maximum number of results to return.
        company:     Case-insensitive ticker filter, e.g. ``"AAPL"``.
        quarter:     Case-insensitive quarter filter, e.g. ``"Q1"``.
        year:        Year filter, e.g. ``"2024"``.
        persist_dir: Path to the saved FAISS index directory.

    Returns:
        List of up to *k* :class:`Document` objects with source metadata.
    """
    from embedder import load_vectorstore

    vectorstore = load_vectorstore(persist_dir)

    # Fetch an oversized candidate pool when filters are active
    fetch_k = k * 4 if (company or quarter or year) else k
    results: List[Document] = vectorstore.similarity_search(query, k=fetch_k)

    # Apply metadata filters (case-insensitive for strings)
    if company:
        results = [
            d for d in results
            if d.metadata.get("company", "").upper() == company.upper()
        ]
    if quarter:
        results = [
            d for d in results
            if d.metadata.get("quarter", "").upper() == quarter.upper()
        ]
    if year:
        results = [
            d for d in results
            if str(d.metadata.get("year", "")) == str(year)
        ]

    return results[:k]


def format_sources(docs: List[Document]) -> str:
    """
    Format a list of retrieved documents as a human-readable citation string.

    Args:
        docs: List of :class:`Document` objects with metadata.

    Returns:
        Multi-line string with one citation per unique source+page pair.
    """
    seen = set()
    lines = []
    for doc in docs:
        citation = (
            f"  • {doc.metadata.get('source', 'unknown')} "
            f"(p.{doc.metadata.get('page', '?')}) "
            f"— {doc.metadata.get('company', '')} "
            f"{doc.metadata.get('quarter', '')} "
            f"{doc.metadata.get('year', '')}"
        )
        if citation not in seen:
            seen.add(citation)
            lines.append(citation)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Script usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "What was the revenue last quarter?"
    results = similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source : {doc.metadata}")
        print(doc.page_content[:300])
