# Research Analyst

An agentic research & report analyst built with **RAG, LangChain, LangGraph** and classic **ML** — using only free tools.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 0 | Setup: local LLM (Ollama), project skeleton, smoke test | ⏳ |
| 1 | Basic RAG: PDF ingestion, local embeddings, Qdrant, Streamlit UI | ⏳ |
| 2 | Better RAG: hybrid search (BM25 + vector), re-ranker, smart chunking | ✅ |
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

## Stage 1 — Chat with your documents

```bash
# Index documents (PDF / .md / .txt). Default folder: data/docs
uv run python scripts/ingest.py samples

# Ask from the command line
uv run python scripts/ask.py "Why was Qdrant chosen instead of Chroma?"

# Or use the web UI (upload + chat with citations) at http://localhost:8501
uv run streamlit run app.py
```

How it works:

1. **Load**: PDFs are read page by page (pypdf), so every chunk keeps its page number.
2. **Split**: recursive character splitter, 1000 chars with 150 overlap.
3. **Embed**: `nomic-embed-text` in Ollama (768-dim), with the `search_document:` / `search_query:` prefixes the model expects.
4. **Store**: Qdrant in local mode (`data/qdrant`), no server. Chunk ids are deterministic, so re-indexing a file replaces its old chunks.
5. **Answer**: top-4 chunks go to the LLM as numbered passages; it must cite them like `[1]` or say it couldn't find the answer.

> Qdrant local mode locks its folder: stop the Streamlit app before running `scripts/ingest.py` or `scripts/ask.py`.

## Stage 2 — Better retrieval

```bash
# Rebuild the index after upgrading from Stage 1 (adds BM25 vectors)
uv run python scripts/ingest.py --reset samples data/docs

# Compare search modes from the command line
uv run python scripts/ask.py --mode dense --no-rerank "How many dimensions do the vectors have?"
uv run python scripts/ask.py --mode hybrid "How many dimensions do the vectors have?"
```

- **Hybrid search**: each chunk is stored with a dense vector (meaning) and a sparse BM25 vector (exact keywords; Qdrant applies IDF). Hybrid mode fuses both result lists with Reciprocal Rank Fusion inside Qdrant. Modes: `dense`, `sparse`, `hybrid` (default).
- **Re-ranking**: the top 20 fused candidates are re-scored by a cross-encoder (`ms-marco-MiniLM-L-12-v2` via FlashRank, ONNX on CPU, ~34 MB, ~0.1 s) and the best 4 are kept.
- **Section-aware chunking**: Markdown is split on headings first, so a chunk never mixes two sections; continuation chunks are prefixed with their section path (`[Guide > Setup]`) so they stay findable. Citations show the section (`handbook.md › Setup`) or PDF page.
- Search mode and re-ranking can be toggled in the app sidebar.

### Troubleshooting

- **`llama-server process has terminated: exit status 0xc0000005`**: Ollama tried to use an old/unsupported GPU (e.g. GeForce GT 730 via Vulkan). Force CPU by setting the user environment variables `GGML_VK_VISIBLE_DEVICES=-1` and `CUDA_VISIBLE_DEVICES=-1`, then restart Ollama.
- **Model download keeps stalling**: `ollama pull` opens 16 parallel connections, which flaky connections can't sustain. Download the model files with a single resumable stream instead and drop them into `OLLAMA_MODELS/blobs` with the manifest under `manifests/registry.ollama.ai/library/<model>/<tag>`. Set `OLLAMA_NOPRUNE=1` so Ollama doesn't delete them half-way.

### Switching LLM provider

All code calls `research_analyst.llm.get_llm()`; the provider is chosen in `.env`:

| Provider | `.env` | Extra install |
|---|---|---|
| Ollama (local, default) | `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.2:3b` | — |
| Groq (free tier) | `LLM_PROVIDER=groq`, `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=...` | `uv sync --extra groq` |
| Gemini (free tier) | `LLM_PROVIDER=google_genai`, `LLM_MODEL=gemini-2.5-flash`, `GOOGLE_API_KEY=...` | `uv sync --extra gemini` |

## Project layout

```
app.py                  Streamlit UI (upload + chat)
src/research_analyst/   application package
  config.py             settings from .env
  llm.py                provider-agnostic LLM factory
  embeddings.py         Ollama embeddings with nomic task prefixes
  vectorstore.py        Qdrant local-mode store (dense + BM25 vectors)
  retrieval.py          dense / sparse / hybrid search + re-ranking
  ingest.py             load → section-aware split → index
  rag.py                retrieve → answer with citations
  ui.py                 shared Streamlit helpers
scripts/                runnable entry points (hello_llm, ingest, ask)
samples/                sample document used by tests
tests/                  pytest suite (Ollama tests auto-skip if models are missing)
data/                   local documents + vector DB (git-ignored)
notebooks/              experiments
```
