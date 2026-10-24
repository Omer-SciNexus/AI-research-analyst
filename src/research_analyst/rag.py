"""Basic RAG: retrieve relevant chunks, then answer with inline citations."""

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

from research_analyst.llm import get_llm
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


def stream_answer(question: str, docs: list[Document]) -> Iterator[str]:
    if not docs:
        yield NOT_FOUND
        return
    chain = PROMPT | get_llm()
    config = {"callbacks": get_callbacks(), "run_name": "document_chat"}
    for chunk in chain.stream({"context": format_context(docs), "question": question}, config=config):
        yield chunk.content


def answer(question: str, store: QdrantVectorStore) -> tuple[str, list[Document]]:
    docs = retrieve(question, store)
    return "".join(stream_answer(question, docs)), docs
