"""Local embeddings served by Ollama."""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from research_analyst.config import settings


class PrefixedEmbeddings(Embeddings):
    """Adds the task prefixes nomic-embed-text was trained with.

    nomic-embed-text expects "search_document: " on indexed text and
    "search_query: " on queries; without them retrieval quality drops.
    Other models are passed through unchanged.
    """

    def __init__(self, model: str, base_url: str):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._use_prefix = model.startswith("nomic-embed-text")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._use_prefix:
            texts = [f"search_document: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._use_prefix:
            text = f"search_query: {text}"
        return self._inner.embed_query(text)


def get_embeddings() -> Embeddings:
    return PrefixedEmbeddings(settings.embedding_model, settings.ollama_base_url)
