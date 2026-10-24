"""Qdrant vector store in local (embedded) mode - no server needed.

Every chunk is stored with two vectors: a dense one (nomic-embed-text) and a
sparse BM25 one, so the same collection serves dense, keyword and hybrid search.

Local mode locks its folder, so only one process (the Streamlit app OR a CLI
script) can open data/qdrant at a time.
"""

from functools import lru_cache

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from research_analyst.config import settings
from research_analyst.embeddings import get_embeddings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
MODES = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=settings.sparse_model, cache_dir=str(settings.models_dir))


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    dim = len(get_embeddings().embed_query("dimension probe"))
    client.create_collection(
        collection_name,
        vectors_config={DENSE_VECTOR: VectorParams(size=dim, distance=Distance.COSINE)},
        # IDF modifier turns raw term frequencies into proper BM25 scores
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
    )


def build_vectorstore(client: QdrantClient, collection_name: str, mode: str = "hybrid") -> QdrantVectorStore:
    if not client.collection_exists(collection_name):
        _create_collection(client, collection_name)
    else:
        params = client.get_collection(collection_name).config.params
        if not params.sparse_vectors:
            raise RuntimeError(
                f"Collection '{collection_name}' was built before hybrid search was added. "
                "Rebuild it with: uv run python scripts/ingest.py --reset"
            )
        dim = len(get_embeddings().embed_query("dimension probe"))
        if params.vectors[DENSE_VECTOR].size != dim:
            raise RuntimeError(
                f"Collection '{collection_name}' was built with {params.vectors[DENSE_VECTOR].size}-dim embeddings "
                f"but {settings.embedding_model} produces {dim}. Rebuild it with: "
                "uv run python scripts/ingest.py --reset"
            )
    return _view(client, collection_name, mode)


def _view(client: QdrantClient, collection_name: str, mode: str, validate: bool = True) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=MODES[mode],
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
        validate_collection_config=validate,
    )


def with_mode(store: QdrantVectorStore, mode: str) -> QdrantVectorStore:
    """The same collection, searched in another mode (dense / sparse / hybrid)."""
    return _view(store.client, store.collection_name, mode, validate=False)


def get_vectorstore(reset: bool = False) -> QdrantVectorStore:
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.vector_db_path))
    if reset and client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    return build_vectorstore(client, settings.collection_name)


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))])


def delete_source(store: QdrantVectorStore, source: str) -> None:
    store.client.delete(store.collection_name, points_selector=_source_filter(source))


def list_sources(store: QdrantVectorStore) -> dict[str, int]:
    """Return {source file name: number of chunks} for everything indexed."""
    counts: dict[str, int] = {}
    offset = None
    while True:
        points, offset = store.client.scroll(
            store.collection_name, limit=256, offset=offset, with_payload=["metadata"]
        )
        for point in points:
            source = (point.payload or {}).get("metadata", {}).get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        if offset is None:
            return counts
