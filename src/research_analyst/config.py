"""Settings loaded from environment variables / .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")
# Windows without Developer Mode can't symlink; the HF cache still works, just warns loudly.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # RAG
    docs_dir: Path = _path("DOCS_DIR", "data/docs")
    vector_db_path: Path = _path("VECTOR_DB_PATH", "data/qdrant")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Stage 2: hybrid search + re-ranking
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | sparse | hybrid
    rerank: bool = os.getenv("RERANK", "true").lower() in {"1", "true", "yes"}
    fetch_k: int = int(os.getenv("FETCH_K", "20"))  # candidates handed to the re-ranker
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    reranker_model: str = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
    models_dir: Path = _path("MODELS_DIR", "data/models")


settings = Settings()
