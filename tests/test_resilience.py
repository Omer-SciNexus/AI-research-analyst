"""The Ollama server can be restarted mid-request (e.g. crashed by a VPN's network hook)."""

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from research_analyst import rag

DOCS = [Document(page_content="The chunk size is 1000.", metadata={"source": "h.md"})]


def flaky_llm(fail_on_calls: set[int], fail_after_first_chunk: bool = False):
    calls = {"n": 0}

    def generate(inputs):
        for _ in inputs:
            pass
        calls["n"] += 1
        if calls["n"] in fail_on_calls and not fail_after_first_chunk:
            raise httpx.ReadError("connection forcibly closed")
        yield AIMessageChunk(content="It is 1000 ")
        if calls["n"] in fail_on_calls:
            raise httpx.ReadError("connection forcibly closed")
        yield AIMessageChunk(content="[1].")

    return RunnableGenerator(generate), calls


@pytest.fixture(autouse=True)
def no_tracing_or_waiting(monkeypatch):
    monkeypatch.setattr(rag, "get_callbacks", lambda: [])
    monkeypatch.setattr(rag.time, "sleep", lambda seconds: None)


def test_stream_retries_when_connection_drops_before_first_token(monkeypatch):
    llm, calls = flaky_llm(fail_on_calls={1, 2})
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    assert "".join(rag.stream_answer("q", DOCS)) == "It is 1000 [1]."
    assert calls["n"] == 3


def test_stream_does_not_repeat_text_after_partial_output(monkeypatch):
    llm, calls = flaky_llm(fail_on_calls={1}, fail_after_first_chunk=True)
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    received = []
    with pytest.raises(httpx.ReadError):
        for chunk in rag.stream_answer("q", DOCS):
            received.append(chunk)
    assert received == ["It is 1000 "] and calls["n"] == 1


def test_generate_answer_retries(monkeypatch):
    llm, calls = flaky_llm(fail_on_calls={1})
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    monkeypatch.setattr(rag, "with_retry", lambda r: r.with_retry(
        retry_if_exception_type=(httpx.TransportError,), stop_after_attempt=3, wait_exponential_jitter=False
    ))
    assert rag.generate_answer("q", DOCS) == "It is 1000 [1]."
    assert calls["n"] == 2
