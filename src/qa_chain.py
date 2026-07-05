"""
qa_chain.py — QA Chain for the RAG Document Q&A System.

Builds a LangChain ``RetrievalQA`` chain that combines the FAISS retriever
with an OpenAI LLM under a strict financial analyst system prompt.

Includes ``ask_vanilla_llm`` for side-by-side hallucination comparison.
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TOP_K,
    VECTORSTORE_DIR,
)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm():
    """
    Instantiate and return the configured OpenAI chat model.

    Reads ``OPENAI_API_KEY`` from the environment; ``python-dotenv`` should
    have already loaded ``.env`` before this is called.

    Returns:
        A :class:`~langchain_openai.ChatOpenAI` instance.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is absent from the environment.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def _import_retrieval_qa_class():
    """Support both older and newer LangChain package layouts for RetrievalQA."""
    try:
        from langchain.chains import RetrievalQA as RetrievalQAClass
        return RetrievalQAClass
    except ImportError:
        try:
            from langchain.chains.retrieval_qa import RetrievalQA as RetrievalQAClass
            return RetrievalQAClass
        except ImportError:
            from langchain.chains.retrieval_qa.base import RetrievalQA as RetrievalQAClass
            return RetrievalQAClass


def build_qa_chain(
    k: int = TOP_K,
    persist_dir: Path = VECTORSTORE_DIR,
):
    """
    Construct a ``RetrievalQA`` chain with the financial analyst system prompt.

    The chain uses the "stuff" strategy: all retrieved chunks are concatenated
    into a single context window before the LLM generates an answer.

    Args:
        k:           Number of chunks to retrieve per query.
        persist_dir: Path to the persisted FAISS vectorstore.

    Returns:
        A configured :class:`~langchain.chains.RetrievalQA` instance.
    """
    RetrievalQA = _import_retrieval_qa_class()
    from langchain.prompts import PromptTemplate
    from retriever import get_retriever

    # Inject the system prompt as the preamble of the RAG prompt template
    prompt_template = (
        f"{SYSTEM_PROMPT}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = get_llm()
    retriever = get_retriever(persist_dir=persist_dir, k=k)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def ask(
    query: str,
    qa_chain=None,
    k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Submit a natural language question to the RAG pipeline.

    Args:
        query:    The question to answer.
        qa_chain: An existing ``RetrievalQA`` chain.  A new one is built
                  automatically when ``None``.
        k:        Number of chunks to retrieve (ignored if *qa_chain* is
                  supplied).

    Returns:
        Dict with:

        - ``"answer"``        – generated answer string
        - ``"sources"``       – list of source :class:`Document` objects
        - ``"source_summary"``– deduplicated list of ``"file (p.N)"`` strings
    """
    if qa_chain is None:
        qa_chain = build_qa_chain(k=k)

    result = qa_chain.invoke({"query": query})

    source_docs: List = result.get("source_documents", [])
    # Deduplicate citations while preserving order
    seen: set = set()
    source_summary: List[str] = []
    for doc in source_docs:
        citation = (
            f"{doc.metadata.get('source', 'unknown')} "
            f"(p.{doc.metadata.get('page', '?')})"
        )
        if citation not in seen:
            seen.add(citation)
            source_summary.append(citation)

    return {
        "answer": result["result"],
        "sources": source_docs,
        "source_summary": source_summary,
    }


def ask_vanilla_llm(query: str) -> str:
    """
    Ask the same question to the LLM **without** any retrieval context.

    Used in the notebook's Hallucination Check section to contrast RAG
    answers against unconstrained generation.

    Args:
        query: The question to ask.

    Returns:
        The raw model response as a string.
    """
    from langchain.schema import HumanMessage, SystemMessage

    llm = get_llm()
    messages = [
        SystemMessage(content="You are a financial analyst assistant."),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return response.content


def pretty_print(result: Dict[str, Any]) -> None:
    """
    Print a formatted answer and source citations to stdout.

    Args:
        result: Dict returned by :func:`ask`.
    """
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])
    print("\nSOURCES")
    print("-" * 70)
    for citation in result["source_summary"]:
        print(f"  • {citation}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Script usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chain = build_qa_chain()
    result = ask("What was Apple's revenue in Q1 2024?", qa_chain=chain)
    pretty_print(result)
