import urllib.request

import pytest

from research_analyst.config import settings
from research_analyst.llm import get_llm


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(settings.ollama_base_url, timeout=2)
        return True
    except OSError:
        return False


def test_get_llm_builds_ollama_model():
    llm = get_llm(provider="ollama", model="llama3.2:3b")
    assert llm.model == "llama3.2:3b"


@pytest.mark.skipif(not _ollama_running(), reason="Ollama server not running")
def test_llm_answers():
    reply = get_llm(provider="ollama").invoke("Reply with exactly one word: pong")
    assert "pong" in reply.content.lower()
