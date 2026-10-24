"""Qdrant vector store in local (embedded) mode - no server needed.

Local mode locks its folder, so only one process (the Streamlit app OR a CLI
script) can open data/qdrant at a time.
"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from research_analyst.config import settings
from research_analyst.embeddings import get_embeddings


def build_vectorstore(client: QdrantClient, collection_name: str) -> QdrantVectorStore:
    embeddings = get_embeddings()
    if not client.collection_exists(collection_name):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)


def get_vectorstore() -> QdrantVectorStore:
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.vector_db_path))
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
