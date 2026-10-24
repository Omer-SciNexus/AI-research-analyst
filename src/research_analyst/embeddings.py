"""Local embeddings served by Ollama."""

import httpx
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from research_analyst.config import settings

# The Ollama app occasionally restarts its server (e.g. after an update check);
# a dropped connection is retried instead of failing a whole ingest or evaluation.
_retry = retry(
    retry=retry_if_exception_type((httpx.TransportError, ConnectionError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)


class PrefixedEmbeddings(Embeddings):
    """Adds the task prefixes nomic-embed-text was trained with.

    nomic-embed-text expects "search_document: " on indexed text and
    "search_query: " on queries; without them retrieval quality drops.
    Other models are passed through unchanged.
    """

    def __init__(self, model: str, base_url: str):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._use_prefix = model.startswith("nomic-embed-text")

    @_retry
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._use_prefix:
            texts = [f"search_document: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    @_retry
    def embed_query(self, text: str) -> list[float]:
        if self._use_prefix:
            text = f"search_query: {text}"
        return self._inner.embed_query(text)


def get_embeddings() -> Embeddings:
    return PrefixedEmbeddings(settings.embedding_model, settings.ollama_base_url)
