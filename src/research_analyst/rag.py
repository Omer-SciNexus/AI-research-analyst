"""Basic RAG: retrieve relevant chunks, then answer with inline citations."""

import time
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

from research_analyst.confidence import assess
from research_analyst.config import settings
from research_analyst.llm import RETRYABLE_ERRORS, get_llm, with_retry
from research_analyst.retrieval import search
from research_analyst.tracing import get_callbacks

NOT_FOUND = "I couldn't find this in the indexed documents."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful research assistant. Answer the question using ONLY the "
            "numbered context passages. Cite the passages you use inline, like [1] or [2][3]. "
            f'If the passages do not contain the answer, reply exactly: "{NOT_FOUND}" '
            "Be concise.",
        ),
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]
)


def source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    if page := doc.metadata.get("page"):
        return f"{source}, p.{page}"
    if section := doc.metadata.get("section"):
        return f"{source} › {section.split(' > ')[-1]}"
    return source


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({source_label(doc)})\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )


def retrieve(question: str, store: QdrantVectorStore, **search_options) -> list[Document]:
    """Hybrid search + re-ranking by default; see retrieval.search for options."""
    return search(question, store, **search_options)


@dataclass
class PreparedTurn:
    docs: list[Document]
    confidence: float | None  # P(documents can answer), None if no model is trained
    threshold: float | None
    skip_llm: bool  # confidence too low: reply "not found" without calling the LLM


def prepare(
    question: str,
    store: QdrantVectorStore,
    mode: str | None = None,
    use_rerank: bool | None = None,
    skip_low_confidence: bool | None = None,
) -> PreparedTurn:
    """Retrieve passages and score answerability (Stage 5) before any LLM call."""
    assessment = assess(question, store)  # hybrid + re-rank search, the default setup
    default_setup = (mode or settings.retrieval_mode) == "hybrid" and (settings.rerank if use_rerank is None else use_rerank)
    docs = assessment.docs if default_setup else retrieve(question, store, mode=mode, use_rerank=use_rerank)
    skip = settings.skip_low_confidence if skip_low_confidence is None else skip_low_confidence
    return PreparedTurn(
        docs=docs,
        confidence=assessment.confidence,
        threshold=assessment.threshold,
        skip_llm=skip and not assessment.answerable,
    )


def _inputs(question: str, docs: list[Document]) -> dict:
    return {"context": format_context(docs), "question": question}


def stream_answer(question: str, docs: list[Document], attempts: int = 4) -> Iterator[str]:
    """Stream the answer token by token (for the UI).

    A dropped connection before the first token is retried; once text has been shown,
    retrying would repeat it, so the error is raised instead.
    """
    if not docs:
        yield NOT_FOUND
        return
    chain = PROMPT | get_llm()
    config = {"callbacks": get_callbacks(), "run_name": "document_chat"}
    for attempt in range(1, attempts + 1):
        started = False
        try:
            for chunk in chain.stream(_inputs(question, docs), config=config):
                started = True
                yield chunk.content
            return
        except RETRYABLE_ERRORS:
            if started or attempt == attempts:
                raise
            time.sleep(2 * attempt)


def generate_answer(question: str, docs: list[Document]) -> str:
    """Whole answer at once, retried if the LLM server drops the connection."""
    if not docs:
        return NOT_FOUND
    config = {"callbacks": get_callbacks(), "run_name": "document_chat"}
    return with_retry(PROMPT | get_llm()).invoke(_inputs(question, docs), config=config).content


def answer(question: str, store: QdrantVectorStore) -> tuple[str, list[Document]]:
    docs = retrieve(question, store)
    return generate_answer(question, docs), docs
