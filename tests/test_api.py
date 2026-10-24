"""API tests, fully offline: fake embeddings, fake LLM and the fake research agent."""

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient
from test_agent import FakeAgent

from research_analyst import api, confidence, rag, vectorstore
from research_analyst.ingest import ingest_file

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "rag_handbook.md"


class StubModel:
    name, threshold = "stub", 0.5

    def __init__(self, p: float):
        self.p = p

    def predict(self, features):
        return self.p


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(confidence, "load_model", lambda: None)
    monkeypatch.setattr(api, "load_model", lambda: None)
    monkeypatch.setattr(api, "generate_answer", lambda q, docs: f"Answer to {q} [1].")
    monkeypatch.setattr(api, "stream_answer", lambda q, docs: iter(["Answer ", "[1]."]))
    monkeypatch.setattr(api, "settings", dataclasses.replace(api.settings, docs_dir=tmp_path))

    store = vectorstore.build_vectorstore(QdrantClient(":memory:"), "api")
    ingest_file(SAMPLE, store)
    api.app.state.store = store
    api.app.state.graph = FakeAgent(verdicts=[True] * 3).build(InMemorySaver())
    with TestClient(api.app) as c:
        yield c


def test_health_and_index_page(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["chunks"] == 7 and health["confidence_model"] is None
    assert "Research Analyst" in client.get("/").text


def test_upload_list_and_delete_documents(client):
    files = [("files", ("notes.md", b"# Notes\n\nThe launch date is 12 March.", "text/markdown"))]
    assert client.post("/api/documents", files=files).json() == {"notes.md": 1}
    assert client.get("/api/documents").json()["notes.md"] == 1
    assert client.delete("/api/documents/notes.md").json() == {"removed": "notes.md"}
    assert client.delete("/api/documents/notes.md").status_code == 404


def test_upload_rejects_unsupported_files_and_paths(client):
    bad = client.post("/api/documents", files=[("files", ("run.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415
    sneaky = client.post("/api/documents", files=[("files", ("../../evil.md", b"# x\n\ntext", "text/markdown"))])
    assert sneaky.json() == {"evil.md": 1}  # directory parts are dropped


def test_ask_returns_answer_sources_and_confidence(client):
    body = client.post("/api/ask", json={"question": "What chunk size is used?"}).json()
    assert body["answer"] == "Answer to What chunk size is used? [1]."
    assert body["confidence"] is None and body["llm_skipped"] is False
    assert len(body["sources"]) == 4 and body["sources"][0]["label"].startswith("rag_handbook.md")


def test_low_confidence_skips_the_llm(client, monkeypatch):
    monkeypatch.setattr(confidence, "load_model", lambda: StubModel(0.1))
    monkeypatch.setattr(api, "generate_answer", lambda q, d: pytest.fail("LLM should be skipped"))
    body = client.post("/api/ask", json={"question": "How do I bake bread?"}).json()
    assert body["llm_skipped"] and body["answer"] == rag.NOT_FOUND and body["confidence"] == 0.1


def test_ask_stream_emits_meta_tokens_done(client):
    response = client.post("/api/ask/stream", json={"question": "What chunk size is used?"})
    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1].removeprefix("data: ")))
        for block in response.text.strip().split("\n\n")
    ]
    assert [e for e, _ in events] == ["meta", "token", "token", "done"]
    assert len(events[0][1]["sources"]) == 4
    assert "".join(d for e, d in events if e == "token") == "Answer [1]."


def test_research_flow_with_approval(client):
    started = client.post("/api/research", json={"question": "What is X?", "web": "off"}).json()
    assert started["status"] == "awaiting_approval" and started["plan"] == ["What is X?", "Why does X matter?"]

    run = started["thread_id"]
    done = client.post(f"/api/research/{run}/resume", json={"action": "approve"}).json()
    assert done["status"] == "done" and "## Summary" in done["report"] and done["sources"]

    assert client.post(f"/api/research/{run}/resume", json={"action": "approve"}).status_code == 409
    assert client.get(f"/api/research/{run}").json()["status"] == "done"
    assert client.get("/api/research/does-not-exist").status_code == 404


def test_validation_errors(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/research", json={"question": "x", "web": "sometimes"}).status_code == 422
