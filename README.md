# Research Analyst

An agentic research & report analyst built with **RAG, LangChain, LangGraph** and classic **ML** — using only free tools.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 0 | Setup: local LLM (Ollama), project skeleton, smoke test | ⏳ |
| 1 | Basic RAG: PDF ingestion, local embeddings, Chroma, Streamlit UI | |
| 2 | Better RAG: hybrid search (BM25 + vector), re-ranker, smart chunking | |
| 3 | LangGraph agents: planner / researcher / fact-checker / writer, human-in-the-loop | |
| 4 | Evaluation & observability: RAGAS, Langfuse | |
| 5 | ML: fine-tuned embeddings / query router / confidence scorer | |
| 6 | Product: FastAPI + frontend, deploy to Hugging Face Spaces | |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
# 1. Pull local models (stored in %OLLAMA_MODELS%)
ollama pull llama3.2:3b        # fast chat model (~2 GB)
ollama pull nomic-embed-text   # embeddings for RAG (~0.3 GB)
ollama pull qwen2.5:7b         # optional: better quality (~4.7 GB)

# 2. Install Python deps (creates .venv with Python 3.12)
uv sync

# 3. Configure
cp .env.example .env           # then edit LLM_PROVIDER / LLM_MODEL if needed

# 4. Smoke test
uv run python scripts/hello_llm.py "What is LangGraph?"
uv run pytest
```

### Switching LLM provider

All code calls `research_analyst.llm.get_llm()`; the provider is chosen in `.env`:

| Provider | `.env` | Extra install |
|---|---|---|
| Ollama (local, default) | `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.2:3b` | — |
| Groq (free tier) | `LLM_PROVIDER=groq`, `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=...` | `uv sync --extra groq` |
| Gemini (free tier) | `LLM_PROVIDER=google_genai`, `LLM_MODEL=gemini-2.5-flash`, `GOOGLE_API_KEY=...` | `uv sync --extra gemini` |

## Project layout

```
src/research_analyst/   application package
  config.py             settings from .env
  llm.py                provider-agnostic LLM factory
scripts/                runnable entry points
tests/                  pytest suite
data/                   local documents (git-ignored)
notebooks/              experiments
```
