# Research Analyst

An agentic research & report analyst built with **RAG, LangChain, LangGraph** and classic **ML** — using only free tools.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
# 1. Pull local models
ollama pull llama3.2:3b        # chat model (~2 GB)
ollama pull nomic-embed-text   # embeddings (~0.3 GB)

# 2. Install Python deps
uv sync

# 3. Configure
cp .env.example .env           # then edit LLM_PROVIDER / LLM_MODEL if needed

# 4. Download voice models (~550 MB, resumable: re-run if it stops)
bash scripts/download_voice_models.sh

# 5. Smoke test
uv run python scripts/hello_llm.py "What is LangGraph?"
uv run pytest
```

To use a free cloud LLM instead of Ollama, set it in `.env`:

| Provider | `.env` | Extra install |
|---|---|---|
| Groq | `LLM_PROVIDER=groq`, `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=...` | `uv sync --extra groq` |
| Gemini | `LLM_PROVIDER=google_genai`, `LLM_MODEL=gemini-2.5-flash`, `GOOGLE_API_KEY=...` | `uv sync --extra gemini` |

## Run

> The Streamlit app, the API and the scripts below share `data/qdrant`, which only one process can open: stop the app before running a script.

**Document chat** (upload, cited answers, voice) at http://localhost:8501:

```bash
uv run streamlit run app.py
```

Voice: record a question in Korean with the mic above the chat box; it is translated to English and answered. Turn on **Speak answers** in the sidebar to hear the reply. Set `STT_LANGUAGE` in `.env` to speak another language (empty = auto-detect).

**Index documents and ask from the command line:**

```bash
uv run python scripts/ingest.py samples
uv run python scripts/ask.py "Why was Qdrant chosen instead of Chroma?"
uv run python scripts/ask.py --mode dense --no-rerank "How many dimensions do the vectors have?"
```

**Research agent** (also the "Research Agent" page in the app):

```bash
uv run python scripts/research.py "How does hybrid search improve RAG?"
uv run python scripts/research.py --web "What is LangGraph used for?"
```

**Evaluation** (results on the "Evaluation" page in the app):

```bash
uv run python scripts/evaluate.py             # retrieval metrics
uv run python scripts/evaluate.py --answers   # + answer quality
```

**Confidence model** (retrain):

```bash
uv run python scripts/make_ml_dataset.py
uv run python scripts/train_confidence.py
```

**API + web UI** at http://localhost:8000 (API docs at `/docs`):

```bash
uv run uvicorn research_analyst.api:app --port 8000
```

**Deploy to Hugging Face Spaces:**

```bash
set HF_TOKEN=hf_...
set GROQ_API_KEY=gsk_...
uv run python scripts/deploy_space.py --space <your-username>/research-analyst
```
