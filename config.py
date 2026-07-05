"""
config.py — Centralized configuration for the RAG Document Q&A System.

All tunable parameters live here. Edit this file to change model names,
paths, chunking behavior, and retrieval settings without touching source code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "reports"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ---------------------------------------------------------------------------
# PDF Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# Set USE_OPENAI_EMBEDDINGS = True to swap HuggingFace for OpenAI embeddings.
USE_OPENAI_EMBEDDINGS = False
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5   # number of chunks returned per query

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o"         # or "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.0        # 0 = deterministic; increase for creativity
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer questions strictly based "
    "on the provided context from quarterly earnings reports. Always cite the "
    "source document and page number. If the answer is not in the context, "
    "say 'I could not find this information in the provided reports.' "
    "Do not hallucinate financial figures."
)
