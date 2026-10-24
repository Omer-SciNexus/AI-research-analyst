# Research Analyst

An agentic research & report analyst built with **RAG, LangChain, LangGraph** and classic **ML** — using only free tools.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 0 | Setup: local LLM (Ollama), project skeleton, smoke test | ✅ |
| 1 | Basic RAG: PDF ingestion, local embeddings, Qdrant, Streamlit UI | ✅ |
| 2 | Better RAG: hybrid search (BM25 + vector), re-ranker, smart chunking | ✅ |
| 3 | LangGraph agents: planner / researcher / fact-checker / writer, human-in-the-loop | ✅ |
| 4 | Evaluation & observability: retrieval/answer metrics, LLM judge, local tracing | ✅ |
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

## Stage 3 — Research agent (LangGraph)

```bash
uv run python scripts/research.py "How does hybrid search improve RAG?"
uv run python scripts/research.py --web "What is LangGraph used for?"
# or open the "Research Agent" page in the app
```

```
plan ──► review ──► research ×N ──► write ──► fact_check ──► END
 ▲         │ (human)   (parallel)     ▲            │
 └─────────┘ revise                   └────────────┘ unsupported claims (1 retry)
```

| Node | What it does |
|---|---|
| `plan` | LLM splits the question into 2–4 sub-questions (structured output) |
| `review` | **Human-in-the-loop**: the graph pauses with `interrupt()`; approve, edit the sub-questions, or send feedback to re-plan |
| `research` | One parallel task per sub-question (`Send` fan-out): hybrid search + re-rank over your documents, optionally DuckDuckGo snippets, then a cited answer |
| `write` | Merges and de-duplicates all sources into one numbered list, renumbers every citation, adds an executive summary |
| `fact_check` | Validates citation numbers, then an LLM judge checks the summary against the cited passages; unsupported claims send it back to `write` once |

Runs are checkpointed in SQLite (`data/checkpoints.sqlite`), so a run paused for review survives an app restart. Reports are saved to `data/reports/`. With `llama3.2:3b` on CPU a run takes a few minutes; switch `LLM_PROVIDER` to Groq/Gemini for speed.

## Stage 4 — Evaluation & tracing

```bash
uv run python scripts/evaluate.py             # retrieval metrics, ~30 s, no LLM needed
uv run python scripts/evaluate.py --answers   # + answer quality, several minutes on CPU
# results and traces: the "Evaluation" page in the app
```

**Evaluation set** (`samples/eval_set.json`): 15 answerable questions labelled with the section that holds the answer, plus 2 questions the documents can't answer. A distractor document (`samples/rag_glossary.md`) with the same vocabulary but no project facts is indexed alongside, so retrieval has to discriminate. Everything is indexed into a fresh in-memory collection, so runs are reproducible and your own index is untouched.

| Metric | Meaning |
|---|---|
| hit@1 / hit@k | the correct chunk is ranked first / anywhere in the top k |
| MRR | mean of 1 / rank of the correct chunk |
| keyword recall | share of the expected facts that appear in the answer |
| citation rate | answers that cite at least one passage |
| grounding | share of the answer's content words found in the retrieved passages (no LLM) |
| faithfulness | LLM judge (`judge.py`, YES/NO): every claim is supported by the retrieved passages |
| refusal accuracy | unanswerable questions answered with "couldn't find" |

First results (17 questions, top-4, `nomic-embed-text`):

| Setup | hit@1 | hit@4 | MRR | latency |
|---|---|---|---|---|
| dense | 0.80 | 1.00 | 0.90 | 72 ms |
| sparse (BM25) | 0.60 | 0.93 | 0.74 | 36 ms |
| hybrid | 0.67 | 1.00 | 0.82 | 70 ms |
| **hybrid + rerank** | **0.87** | **1.00** | **0.93** | 364 ms |

Re-ranking gives the best ranking. Hybrid alone is slightly behind dense on this small, mostly paraphrased set: RRF fusion lets BM25's weaker matches pull good dense hits down, which the re-ranker then repairs. On corpora heavy in exact names, codes and numbers, BM25's contribution grows.

Answer quality (`llama3.2:3b` on CPU, hybrid + rerank, ~7 s per answer):

| keyword recall | citation rate | grounding | refusal accuracy | faithfulness (3B judge) |
|---|---|---|---|---|
| 0.80 | 1.00 | 0.94 | 1.00 | 0.43 — unreliable, see below |

- All three keyword misses come from the distractor: the right passage was retrieved, but the 3B model answered from a similar glossary passage ("common sizes are 384, 1024, 1536" instead of the project's 768). High grounding shows it rarely invents facts; its weakness is choosing between look-alike passages. A larger model, or keeping only the top 2–3 re-ranked passages, should help.
- **A 3B model is not a reliable judge.** It rejected roughly 10 of 14 answers that a manual check found correct. Small models also do worse with JSON verdicts: a structured `faithful` boolean came back `false` even for exact quotes, which is why `judge.py` asks a one-word YES/NO question (and judges sentence by sentence to find unsupported claims). Point `JUDGE_PROVIDER` / `JUDGE_MODEL` at a larger model (e.g. Groq's free `llama-3.3-70b-versatile`) for trustworthy faithfulness scores and fact-checks; until then, read faithfulness together with grounding.

**Tracing**: a LangChain callback (`tracing.py`) appends every top-level run and every LLM call (node, model, latency, input/output tokens) to `data/traces/<date>.jsonl`. LangGraph passes it down to every node, including parallel research tasks. The Evaluation page lists runs, their LLM calls and generation speed. For a hosted dashboard, Langfuse's free cloud tier plugs in by setting `LANGFUSE_*` in `.env` and running `uv add langfuse`; self-hosting Langfuse needs about 2 GB of Docker images, so it isn't the default.

### Troubleshooting

- **`llama-server process has terminated: exit status 0xc0000005`**: Ollama tried to use an old/unsupported GPU (e.g. GeForce GT 730 via Vulkan). Force CPU by setting the user environment variables `GGML_VK_VISIBLE_DEVICES=-1` and `CUDA_VISIBLE_DEVICES=-1`, then restart Ollama.
- **`[WinError 10054] An existing connection was forcibly closed` / Ollama keeps restarting**: a VPN that hooks Windows networking can crash Ollama. Astrill installs a Winsock LSP (`ASProxy64.dll`, see `netsh winsock show catalog`) that every process loads, in any VPN mode, and Ollama dies inside `WSASend` (exit `0xc0000005`, logged in `%LOCALAPPDATA%\Ollama\server.log`). The tray app restarts the server within seconds and all LLM/embedding calls here retry, so runs survive; streaming chat only retries before the first token. To remove the cause, exclude Ollama from the VPN or remove the LSP (`netsh winsock reset`, which affects the VPN).
- **Ollama app keeps downloading updates**: the tray app downloads each new installer (~1.6 GB) in the background. Turn off auto-update in Ollama's settings (stored as `auto_update_enabled` in `%LOCALAPPDATA%\Ollama\db.sqlite`). `OLLAMA_NO_CLOUD=1` disables its cloud features.
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
app.py                  Streamlit UI: document chat (upload + cited answers)
pages/                  extra app pages (Research Agent, Evaluation & Traces)
src/research_analyst/   application package
  config.py             settings from .env
  llm.py                provider-agnostic LLM factory
  embeddings.py         Ollama embeddings with nomic task prefixes
  vectorstore.py        Qdrant local-mode store (dense + BM25 vectors)
  retrieval.py          dense / sparse / hybrid search + re-ranking
  ingest.py             load → section-aware split → index
  rag.py                retrieve → answer with citations
  agent.py              LangGraph research agent
  web.py                DuckDuckGo search (no API key)
  evaluation.py         retrieval + answer metrics
  judge.py              YES/NO LLM judge (evaluation + agent fact-check)
  tracing.py            JSONL tracer (+ optional Langfuse)
  ui.py                 shared Streamlit helpers
scripts/                runnable entry points (hello_llm, ingest, ask, research, evaluate)
samples/                sample documents + labelled evaluation set
tests/                  pytest suite (Ollama tests auto-skip if models are missing)
data/                   local documents + vector DB (git-ignored)
notebooks/              experiments
```
