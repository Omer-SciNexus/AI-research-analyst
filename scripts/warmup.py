"""Download every local model the app needs, so the first request isn't slow.

Used at Docker build time; also handy before going offline.
    uv run python scripts/warmup.py
"""

from research_analyst.config import settings
from research_analyst.embeddings import get_embeddings
from research_analyst.retrieval import get_reranker
from research_analyst.vectorstore import get_sparse_embeddings


def main() -> None:
    print(f"dense:    {settings.embedding_provider}/{settings.embedding_model}", len(get_embeddings().embed_query("warm up")))
    print(f"sparse:   {settings.sparse_model}", len(get_sparse_embeddings().embed_query("warm up").indices))
    get_reranker()
    print(f"reranker: {settings.reranker_model} ready")


if __name__ == "__main__":
    main()
