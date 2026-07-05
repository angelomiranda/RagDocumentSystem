"""
embedder.py — Embedding & Vector Store for the RAG Document Q&A System.

Embeds chunked documents using either HuggingFace sentence-transformers
(default, offline-capable) or OpenAI embeddings (requires API key), then
persists a FAISS index to disk.  Subsequent runs skip re-embedding if the
index already exists.
"""
import os
import sys
from pathlib import Path
from typing import List

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    USE_OPENAI_EMBEDDINGS,
    HF_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)


# ---------------------------------------------------------------------------
# Embedding model factory
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Instantiate and return the configured embedding model.

    The model is selected by ``USE_OPENAI_EMBEDDINGS`` in ``config.py``:

    - ``False`` (default) → ``HuggingFaceEmbeddings`` using
      ``all-MiniLM-L6-v2`` (runs fully offline).
    - ``True`` → ``OpenAIEmbeddings`` using the model in
      ``OPENAI_EMBEDDING_MODEL`` (requires ``OPENAI_API_KEY`` env var).

    Returns:
        A LangChain :class:`Embeddings` instance.

    Raises:
        EnvironmentError: If OpenAI embeddings are selected but
            ``OPENAI_API_KEY`` is not set.
    """
    if USE_OPENAI_EMBEDDINGS:
        from langchain_openai import OpenAIEmbeddings

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        print(f"[INFO] Embedding model: OpenAI / {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=api_key,
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"[INFO] Embedding model: HuggingFace / {HF_EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Build / Load
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: List[Document],
    persist_dir: Path = VECTORSTORE_DIR,
):
    """
    Embed document chunks and persist a FAISS index to *persist_dir*.

    If the index already exists the function loads and returns the cached
    version without re-embedding (cache-hit check on ``index.faiss``).

    Args:
        chunks: Chunked :class:`Document` objects produced by
            ``pdf_loader.chunk_documents``.
        persist_dir: Directory in which to store the FAISS index files.

    Returns:
        A loaded :class:`~langchain_community.vectorstores.FAISS` instance.

    Raises:
        ValueError: If *chunks* is empty and no cached index exists.
    """
    from langchain_community.vectorstores import FAISS

    index_file = persist_dir / "index.faiss"
    if index_file.exists():
        print(f"[INFO] Cached vectorstore found at '{persist_dir}'. Loading…")
        return load_vectorstore(persist_dir)

    if not chunks:
        raise ValueError(
            "chunks is empty. Run pdf_loader.load_and_chunk_pdfs() first."
        )

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model()

    print(f"[INFO] Embedding {len(chunks)} chunks — this may take a moment…")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(persist_dir))
    print(f"[INFO] Vectorstore saved to '{persist_dir}'.")
    return vectorstore


def load_vectorstore(persist_dir: Path = VECTORSTORE_DIR):
    """
    Load a previously persisted FAISS vectorstore from disk.

    Args:
        persist_dir: Directory containing ``index.faiss`` and
            ``index.pkl``.

    Returns:
        A loaded :class:`~langchain_community.vectorstores.FAISS` instance.

    Raises:
        FileNotFoundError: If no FAISS index is found at *persist_dir*.
    """
    from langchain_community.vectorstores import FAISS

    index_file = persist_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index at '{persist_dir}'. "
            "Call build_vectorstore() first."
        )

    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,   # safe: we wrote this file
    )
    print(f"[INFO] Vectorstore loaded from '{persist_dir}'.")
    return vectorstore


# ---------------------------------------------------------------------------
# Script usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pdf_loader import load_and_chunk_pdfs

    chunks = load_and_chunk_pdfs()
    vs = build_vectorstore(chunks)
    print(f"[INFO] FAISS index dimension: {vs.index.d}")
    print(f"[INFO] Total vectors stored:  {vs.index.ntotal}")
