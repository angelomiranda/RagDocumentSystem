"""
app.py — Streamlit browser interface for the RAG Document Q&A System.

Run with:
    streamlit run app.py

Requires the FAISS vectorstore to have been built first (run the notebook
or call embedder.build_vectorstore() from the Python REPL).
"""
import os
import sys
from pathlib import Path


def _ensure_project_venv_on_path() -> None:
    """Expose the workspace .venv packages to Streamlit even if it launches from another Python env."""
    project_root = Path(__file__).resolve().parent
    venv_lib_dir = project_root / ".venv" / "lib"
    if venv_lib_dir.exists():
        for site_packages in sorted(venv_lib_dir.glob("python*/site-packages")):
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.insert(0, site_packages_str)

    venv_bin_dir = project_root / ".venv" / "bin"
    if venv_bin_dir.exists():
        os.environ["PATH"] = f"{venv_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_ensure_project_venv_on_path()

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & path setup
# ---------------------------------------------------------------------------

load_dotenv()

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import VECTORSTORE_DIR, TOP_K, LLM_MODEL  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider(
        "Chunks retrieved (top-k)",
        min_value=1,
        max_value=15,
        value=TOP_K,
        help="How many document chunks the retriever passes to the LLM.",
    )

    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    st.caption(f"Model: **{LLM_MODEL}**")

    vectorstore_exists = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())
    if vectorstore_exists:
        st.success("Vectorstore loaded ✓")
    else:
        st.error("Vectorstore not found")
        st.caption(
            "Build it first:\n```python\n"
            "from src.embedder import build_vectorstore\n"
            "build_vectorstore()\n```"
        )

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# QA chain — cached so it is only initialised once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading QA chain…")
def load_chain(k: int):
    from qa_chain import build_qa_chain
    return build_qa_chain(k=k)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📊 RAG Document Q&A")
st.caption("Ask questions about your earnings reports. Answers are grounded in the source PDFs.")

if not vectorstore_exists:
    st.warning(
        "No vectorstore found. Build the index first by running the notebook "
        "or calling `build_vectorstore()` from `src/embedder.py`.",
        icon="⚠️",
    )
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY** is not set. Add it to your `.env` file and restart.",
        icon="🔑",
    )
    st.stop()

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for chunk in msg["sources"]:
                    meta = chunk.metadata
                    st.markdown(
                        f"**{meta.get('source', 'unknown')}** — "
                        f"*{meta.get('company', '')} "
                        f"{meta.get('quarter', '')} "
                        f"{meta.get('year', '')}*, "
                        f"page {meta.get('page', '?')}"
                    )
                    st.code(chunk.page_content, language=None)

# Chat input
query = st.chat_input("Ask a question about the earnings reports…")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                chain = load_chain(top_k)
                from qa_chain import ask
                result = ask(query, qa_chain=chain)
                answer = result["answer"]
                sources = result["sources"]
                citation_line = (
                    "\n\n---\n**Sources:** " + " · ".join(result["source_summary"])
                    if result["source_summary"]
                    else ""
                )
            except Exception as exc:
                answer = f"❌ Error: {exc}"
                sources = []
                citation_line = ""

        st.markdown(answer + citation_line)

        if show_sources and sources:
            with st.expander("Sources", expanded=False):
                for chunk in sources:
                    meta = chunk.metadata
                    st.markdown(
                        f"**{meta.get('source', 'unknown')}** — "
                        f"*{meta.get('company', '')} "
                        f"{meta.get('quarter', '')} "
                        f"{meta.get('year', '')}*, "
                        f"page {meta.get('page', '?')}"
                    )
                    st.code(chunk.page_content, language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer + citation_line,
        "sources": sources,
    })
