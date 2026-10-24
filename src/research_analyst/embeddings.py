"""Dense embeddings: Ollama locally, or fastembed (ONNX, no server) for deployment."""

from functools import lru_cache

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


class FastEmbedDense(Embeddings):
    """fastembed's ONNX models (CPU, no server) - used where Ollama isn't available, e.g. a Space."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model, cache_dir=str(settings.models_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class PrefixedEmbeddings(Embeddings):
    """Adds the task prefixes nomic-embed-text was trained with.

    nomic-embed-text expects "search_document: " on indexed text and
    "search_query: " on queries; without them retrieval quality drops.
    Other models are passed through unchanged.
    """

    def __init__(self, inner: Embeddings, use_prefix: bool):
        self._inner = inner
        self._use_prefix = use_prefix

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


@lru_cache
def get_embeddings() -> Embeddings:
    model = settings.embedding_model
    if settings.embedding_provider == "fastembed":
        inner = FastEmbedDense(model)
    else:
        inner = OllamaEmbeddings(model=model, base_url=settings.ollama_base_url, client_kwargs={"timeout": 60})
    return PrefixedEmbeddings(inner, use_prefix="nomic-embed-text" in model)
