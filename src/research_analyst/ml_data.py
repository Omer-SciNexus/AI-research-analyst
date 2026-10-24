"""Stage 5: build the training set for the confidence model.

Answerable questions (label 1) are written by the LLM from each indexed chunk, so
they cover whatever documents you index. Unanswerable ones (label 0) come from
samples/ml_negatives.txt: off-topic questions plus near misses that use the same
vocabulary as the documents. Every question is then turned into retrieval features.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

from research_analyst.config import PROJECT_ROOT
from research_analyst.confidence import extract_features
from research_analyst.llm import get_llm, with_retry

NEGATIVES_FILE = PROJECT_ROOT / "samples" / "ml_negatives.txt"

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write test questions. Write {n} different short questions that can be answered using "
            "only the passage below, each about a specific fact in it. Write one question per line, "
            "without numbering or any other text.",
        ),
        ("human", "{passage}"),
    ]
)


def all_chunks(store: QdrantVectorStore) -> list[Document]:
    chunks, offset = [], None
    while True:
        points, offset = store.client.scroll(store.collection_name, limit=256, offset=offset, with_payload=True)
        chunks += [Document(page_content=p.payload["page_content"], metadata=p.payload.get("metadata", {})) for p in points]
        if offset is None:
            return chunks


def parse_questions(text: str) -> list[str]:
    """Lines that look like questions, with list markers and quotes stripped."""
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if line.endswith("?") and 3 <= len(line.split()) <= 30:
            questions.append(line)
    return questions


def generate_questions(passage: str, n: int = 2) -> list[str]:
    reply = with_retry(QUESTION_PROMPT | get_llm(temperature=0.7)).invoke({"passage": passage[:2000], "n": n})
    return parse_questions(reply.content)[:n]


def load_negatives(path: Path = NEGATIVES_FILE) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def positive_questions(store: QdrantVectorStore, per_chunk: int = 2) -> Iterator[dict]:
    for chunk in all_chunks(store):
        if len(chunk.page_content) < 200:  # headings and fragments make poor questions
            continue
        for question in generate_questions(chunk.page_content, per_chunk):
            yield {"question": question, "label": 1, "origin": chunk.metadata.get("source", "unknown")}


def featurize(rows: list[dict], store: QdrantVectorStore) -> list[dict]:
    return [{**row, "features": extract_features(row["question"], store)[0]} for row in rows]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
