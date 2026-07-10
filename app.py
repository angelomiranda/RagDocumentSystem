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


def resolve_openai_api_key(session_value: str | None = None, env_value: str | None = None) -> str:
    """Return the most relevant API key for the current session."""
    candidate = (session_value or "").strip()
    if candidate:
        return candidate
    candidate = (env_value or "").strip()
    return candidate


def _is_missing_information_answer(answer: str) -> bool:
    """Return True when the model explicitly says it could not find the requested info."""
    normalized_answer = (answer or "").lower()
    missing_information_phrases = (
        "i could not find this information",
        "could not find this information",
        "not found in the provided reports",
        "no relevant information",
    )
    return any(phrase in normalized_answer for phrase in missing_information_phrases)


def should_display_sources(answer: str, sources: list | None = None) -> bool:
    """Return True only when the answer should include source details."""
    if not sources:
        return False
    return not _is_missing_information_answer(answer)


@st.cache_resource(show_spinner="Loading QA chain…")
def load_chain(k: int, api_key_value: str):
    if api_key_value:
        os.environ["OPENAI_API_KEY"] = api_key_value
    from qa_chain import build_qa_chain
    return build_qa_chain(k=k)


def main() -> None:
    """Run the Streamlit UI."""
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------------------------------------------------
    # Sidebar — controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Settings")

        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        st.text_input(
            "OpenAI API Key",
            type="password",
            key="openai_api_key",
            help=(
                "Enter your own OpenAI API key for this session. "
                "Leave it blank to use the key from .env or your deployment secrets."
            ),
        )
        # Non-sensitive indicator showing whether a non-empty OPENAI_API_KEY is available
        env_has_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
        if env_has_key:
            st.caption("OPENAI_API_KEY present in environment: ✓")
        else:
            st.caption("OPENAI_API_KEY present in environment: ✗")

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

    # -----------------------------------------------------------------------
    # Main UI
    # -----------------------------------------------------------------------
    api_key = resolve_openai_api_key(
        session_value=st.session_state.get("openai_api_key", ""),
        env_value=os.environ.get("OPENAI_API_KEY", ""),
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.title("📊 RAG Document Q&A")
    st.caption("Ask questions about your earnings reports. Answers are grounded in the source PDFs.")

    if not vectorstore_exists:
        st.warning(
            "No vectorstore found. Build the index first by running the notebook "
            "or calling `build_vectorstore()` from `src/embedder.py`.",
            icon="⚠️",
        )
        st.stop()

    if not api_key:
        st.error(
            "**OPENAI_API_KEY** is not set. Enter a key in the sidebar or set it in your environment or `.env` file.",
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
            if msg["role"] == "assistant" and show_sources:
                display_sources = msg.get(
                    "display_sources",
                    should_display_sources(msg.get("content", ""), msg.get("sources")),
                )
                if display_sources and msg.get("sources"):
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
                    chain = load_chain(top_k, api_key_value=api_key)
                    from qa_chain import ask
                    result = ask(query, qa_chain=chain)
                    answer = result["answer"]
                    sources = result["sources"]
                    source_summary = result.get("source_summary", [])
                    display_sources = show_sources and should_display_sources(answer, sources)
                    citation_line = (
                        "\n\n---\n**Sources:** " + " · ".join(source_summary)
                        if display_sources and source_summary
                        else ""
                    )
                except Exception as exc:
                    answer = f"❌ Error: {exc}"
                    sources = []
                    citation_line = ""
                    display_sources = False

            st.markdown(answer + citation_line)

            if display_sources and sources:
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
            "display_sources": display_sources,
        })


if __name__ == "__main__":
    main()
