import json
import urllib.request

import pytest

from research_analyst.config import settings
from research_analyst.llm import get_llm


def _ollama_has(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2) as resp:
            return model in {m["name"].removesuffix(":latest") for m in json.load(resp)["models"]}
    except OSError:
        return False


def test_get_llm_builds_ollama_model():
    llm = get_llm(provider="ollama", model="llama3.2:3b")
    assert llm.model == "llama3.2:3b"


@pytest.mark.skipif(not _ollama_has(settings.llm_model), reason=f"{settings.llm_model} not available in Ollama")
def test_llm_answers():
    reply = get_llm(provider="ollama").invoke("Reply with exactly one word: pong")
    assert "pong" in reply.content.lower()
