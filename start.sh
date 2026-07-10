#!/usr/bin/env sh
set -e

# Start script for the Docker container.
# - Builds the FAISS vectorstore if it's missing
# - Then launches Streamlit

APP_DIR=/app
VECTOR_DIR=${APP_DIR}/vectorstore

echo "[start] Checking for vectorstore at ${VECTOR_DIR}"
if [ ! -f "${VECTOR_DIR}/index.faiss" ]; then
  echo "[start] No FAISS index found. Attempting to build vectorstore..."
  # Ensure Python can import from src/
  cd ${APP_DIR}
  python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve() / 'src'))
from pdf_loader import load_and_chunk_pdfs
from embedder import build_vectorstore
chunks = load_and_chunk_pdfs()
if not chunks:
    print('[start] No PDFs found in data/reports/. Skipping vectorstore build.')
else:
    print(f'[start] Loaded {len(chunks)} pages/chunks; building vectorstore...')
    build_vectorstore(chunks)
PY
else
  echo "[start] FAISS index already present. Skipping build."
fi

# Finally start Streamlit
exec streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true
