"""Integration tests - need Ollama running with the models pulled; skipped otherwise."""

import json
import urllib.request
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from research_analyst.config import settings
from research_analyst.ingest import ingest_file
from research_analyst.rag import NOT_FOUND, answer, retrieve
from research_analyst.vectorstore import build_vectorstore, list_sources

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "rag_handbook.md"


def _ollama_models() -> set[str]:
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2) as resp:
            return {m["name"].removesuffix(":latest") for m in json.load(resp)["models"]}
    except OSError:
        return set()


MODELS = _ollama_models()
needs_embeddings = pytest.mark.skipif(
    settings.embedding_model not in MODELS, reason=f"{settings.embedding_model} not available in Ollama"
)
needs_llm = pytest.mark.skipif(settings.llm_model not in MODELS, reason=f"{settings.llm_model} not available in Ollama")


@pytest.fixture(scope="module")
def store():
    store = build_vectorstore(QdrantClient(":memory:"), "test")
    ingest_file(SAMPLE, store)
    return store


@needs_embeddings
def test_reingest_does_not_duplicate(store):
    before = list_sources(store)["rag_handbook.md"]
    ingest_file(SAMPLE, store)
    assert list_sources(store)["rag_handbook.md"] == before


@needs_embeddings
def test_retrieves_relevant_chunk(store):
    docs = retrieve("Why was Qdrant chosen instead of Chroma?", store)
    assert any("Chroma" in d.page_content for d in docs)


@needs_embeddings
@needs_llm
def test_answer_cites_passages(store):
    reply, docs = answer("What chunk size does the project use?", store)
    assert "1000" in reply
    assert "[" in reply and docs


@needs_embeddings
@needs_llm
def test_answer_admits_unknown(store):
    reply, _ = answer("What is the capital city of Australia?", store)
    assert NOT_FOUND.rstrip(".").lower() in reply.lower()
