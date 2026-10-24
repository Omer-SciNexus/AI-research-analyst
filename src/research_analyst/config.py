"""Settings loaded from environment variables / .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")  # ollama | fastembed
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

    # Stage 3: research agent
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300"))  # seconds per LLM request
    checkpoint_db: Path = _path("CHECKPOINT_DB", "data/checkpoints.sqlite")
    reports_dir: Path = _path("REPORTS_DIR", "data/reports")

    # Stage 4: evaluation + tracing
    traces_dir: Path = _path("TRACES_DIR", "data/traces")
    eval_dir: Path = _path("EVAL_DIR", "data/eval")
    eval_set: Path = _path("EVAL_SET", "samples/eval_set.json")
    # LLM-as-judge (evaluation faithfulness + the agent's fact-check). A 3B model is not a
    # reliable judge; point these at a stronger model (e.g. Groq's free tier) when possible.
    judge_provider: str = os.getenv("JUDGE_PROVIDER") or os.getenv("LLM_PROVIDER", "ollama")
    judge_model: str = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "llama3.2:3b")

    # Stage 5: answerability (confidence) model
    ml_dir: Path = _path("ML_DIR", "data/ml")
    confidence_model_path: Path = _path("CONFIDENCE_MODEL", "data/models/confidence.joblib")
    skip_low_confidence: bool = os.getenv("SKIP_LOW_CONFIDENCE", "true").lower() in {"1", "true", "yes"}
    auto_web_fallback: bool = os.getenv("AUTO_WEB_FALLBACK", "true").lower() in {"1", "true", "yes"}


settings = Settings()
