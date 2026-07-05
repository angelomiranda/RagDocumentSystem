"""
pdf_loader.py — PDF Ingestion for the RAG Document Q&A System.

Recursively scans /data/reports/ for PDF files, extracts text page-by-page
using PyMuPDF (fitz), attaches rich metadata, and splits content into
overlapping chunks ready for embedding.

Expected filename format: {TICKER}_{Q#}_{YEAR}.pdf
Example: AAPL_Q1_2024.pdf
"""
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Allow running as a script from the src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename_metadata(filename: str) -> dict:
    """
    Parse company, quarter, and year from a standardised filename.

    Args:
        filename: PDF filename, e.g. ``"AAPL_Q1_2024.pdf"``.

    Returns:
        Dict with keys ``company``, ``quarter``, ``year``.
        Falls back to ``"UNKNOWN"`` for any missing segment.
    """
    stem = Path(filename).stem          # strip .pdf  → "AAPL_Q1_2024"
    parts = stem.split("_")            # ["AAPL", "Q1", "2024"]
    return {
        "company": parts[0] if len(parts) > 0 else "UNKNOWN",
        "quarter": parts[1] if len(parts) > 1 else "UNKNOWN",
        "year":    parts[2] if len(parts) > 2 else "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Core loading
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Open a single PDF and return one :class:`Document` per non-empty page.

    Each document carries metadata:

    - ``source``  – original filename
    - ``company`` – ticker parsed from filename
    - ``quarter`` – e.g. "Q1"
    - ``year``    – e.g. "2024"
    - ``page``    – 1-based page number

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of :class:`~langchain.schema.Document` objects.

    Raises:
        RuntimeError: Wraps any exception from PyMuPDF so callers can skip
            bad files without crashing the pipeline.
    """
    file_meta = parse_filename_metadata(pdf_path.name)
    documents: List[Document] = []

    try:
        with fitz.open(str(pdf_path)) as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                text = page.get_text()
                if not text.strip():
                    continue  # skip blank/image-only pages
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source":  pdf_path.name,
                        "company": file_meta["company"],
                        "quarter": file_meta["quarter"],
                        "year":    file_meta["year"],
                        "page":    page_num,
                    },
                ))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse PDF '{pdf_path.name}': {exc}"
        ) from exc

    return documents


def load_all_pdfs(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Recursively discover and load every ``.pdf`` under *data_dir*.

    Files that fail to parse are logged and skipped; the pipeline continues.

    Args:
        data_dir: Root directory to scan.  Defaults to the path in
            ``config.py``.

    Returns:
        Flat list of page-level :class:`Document` objects from all PDFs.
    """
    pdf_paths = sorted(data_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"[WARNING] No PDF files found in '{data_dir}'.")
        print("          Place .pdf files in data/reports/ and re-run.")
        return []

    all_documents: List[Document] = []
    for pdf_path in pdf_paths:
        print(f"  Loading: {pdf_path.name}")
        try:
            docs = load_pdf(pdf_path)
            all_documents.extend(docs)
        except RuntimeError as err:
            print(f"  [ERROR] {err}")

    print(
        f"\n[INFO] Loaded {len(all_documents)} pages "
        f"from {len(pdf_paths)} PDF(s)."
    )
    return all_documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level documents into overlapping text chunks.

    Chunk size and overlap are read from ``config.py``.  Metadata (source,
    company, quarter, year, page) is propagated to every child chunk.

    Args:
        documents: List of page-level :class:`Document` objects.

    Returns:
        List of chunk-level :class:`Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[INFO] Split {len(documents)} pages → {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )
    return chunks


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def load_and_chunk_pdfs(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    One-stop function: load all PDFs and return chunked documents.

    Args:
        data_dir: Root directory containing PDF files.

    Returns:
        List of chunked :class:`Document` objects, or an empty list if no
        PDFs were found.
    """
    documents = load_all_pdfs(data_dir)
    if not documents:
        return []
    return chunk_documents(documents)


# ---------------------------------------------------------------------------
# Script usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    if chunks:
        sample = chunks[0]
        print(f"\n--- Sample chunk ---")
        print(sample.page_content[:400])
        print(f"\nMetadata: {sample.metadata}")
