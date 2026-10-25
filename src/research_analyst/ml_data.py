"""Stage 5: build the training set for the confidence model.

Answerable questions (label 1) are written by the LLM from each indexed chunk, so
they cover whatever documents you index. Unanswerable ones (label 0) come from
samples/ml_negatives.txt: off-topic questions plus near misses that use the same
vocabulary as the documents. Every question is then turned into retrieval features.

The negatives live in datasets/, not samples/: anything in samples/ gets indexed, and
an indexed copy of the "unanswerable" questions would make them trivially answerable.
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

NEGATIVES_FILE = PROJECT_ROOT / "datasets" / "ml_negatives.txt"

# Questions that copy the passage's wording are too easy (re-rank score ~1.0) and teach the
# model a cut-off real, rephrased questions fall below; so half of them must be paraphrased.
QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write test questions that can be answered using only the passage below, each about a "
            "specific fact in it. Write exactly 2 questions, one per line, without numbering or any "
            "other text. The first may reuse words from the passage. The second must be phrased the way "
            "a user who has never seen the passage would ask, using different words.",
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


def generate_questions(passage: str) -> list[str]:
    """One question close to the passage's wording, one paraphrased."""
    reply = with_retry(QUESTION_PROMPT | get_llm(temperature=0.7)).invoke({"passage": passage[:2000]})
    return parse_questions(reply.content)[:2]


def load_negatives(path: Path = NEGATIVES_FILE) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def positive_questions(store: QdrantVectorStore) -> Iterator[dict]:
    for chunk in all_chunks(store):
        if len(chunk.page_content) < 200:  # headings and fragments make poor questions
            continue
        for style, question in zip(["literal", "paraphrased"], generate_questions(chunk.page_content)):
            yield {"question": question, "label": 1, "origin": chunk.metadata.get("source", "unknown"), "style": style}


def featurize(rows: list[dict], store: QdrantVectorStore) -> list[dict]:
    return [{**row, "features": extract_features(row["question"], store)[0]} for row in rows]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
