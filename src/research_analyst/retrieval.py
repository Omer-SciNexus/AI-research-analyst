"""Stage 2 retrieval: dense / BM25 / hybrid search, optionally re-ranked.

Hybrid search fuses dense and BM25 results with Reciprocal Rank Fusion inside
Qdrant. The re-ranker (a small cross-encoder via FlashRank, ONNX on CPU) then
re-scores the fused candidates against the question and keeps the best top_k.
"""

from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from research_analyst.config import settings
from research_analyst.vectorstore import with_mode


@lru_cache
def get_reranker() -> Ranker:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return Ranker(model_name=settings.reranker_model, cache_dir=str(settings.models_dir), log_level="WARNING")


def rerank(question: str, docs: list[Document], k: int) -> list[Document]:
    if not docs:
        return []
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    results = get_reranker().rerank(RerankRequest(query=question, passages=passages))
    return [
        Document(
            page_content=docs[r["id"]].page_content,
            metadata={**docs[r["id"]].metadata, "rerank_score": round(float(r["score"]), 4)},
        )
        for r in results[:k]
    ]


def search(
    question: str,
    store: QdrantVectorStore,
    mode: str | None = None,
    use_rerank: bool | None = None,
    k: int | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    mode = mode or settings.retrieval_mode
    use_rerank = settings.rerank if use_rerank is None else use_rerank
    k = k or settings.top_k
    fetch_k = max(fetch_k or settings.fetch_k, k) if use_rerank else k

    candidates = with_mode(store, mode).similarity_search(question, k=fetch_k)
    return rerank(question, candidates, k) if use_rerank else candidates
