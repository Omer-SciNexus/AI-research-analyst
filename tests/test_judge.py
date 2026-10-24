"""The judge must accept a faithful answer and reject wrong or partly invented ones (needs the LLM)."""

import pytest
from test_rag import needs_llm

from research_analyst.judge import is_supported, unsupported_claims

pytestmark = needs_llm

PASSAGES = "[1] (h.md)\nEach chunk is turned into a vector by the nomic-embed-text model. The model produces 768-dimensional vectors."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The embedding vectors have 768 dimensions [1].", True),
        ("The embedding vectors have 1536 dimensions [1].", False),
        ("The vectors have 768 dimensions [1] and are stored in Pinecone [1].", False),
    ],
)
def test_judge_verdicts(text, expected):
    assert is_supported(text, PASSAGES) is expected


def test_unsupported_claims_names_the_invented_part():
    claims = unsupported_claims("The vectors have 768 dimensions [1] and are stored in Pinecone [1].", PASSAGES)
    assert any("Pinecone" in c for c in claims)
