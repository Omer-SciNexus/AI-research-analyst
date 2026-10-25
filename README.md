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
| 5 | ML: answerability (confidence) model that skips the LLM or triggers web search | ✅ |
| 6 | Product: FastAPI + web UI, Docker, CI, Hugging Face Spaces deploy script | ✅ (deploy needs your HF token) |

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

## Stage 5 — Confidence model (ML)

```bash
uv run python scripts/make_ml_dataset.py    # LLM writes training questions, ~6 min on CPU
uv run python scripts/train_confidence.py   # cross-validation, hold-out test, saves the model
```

A scikit-learn classifier predicts **whether the indexed documents can answer a question** from retrieval signals alone, before any LLM call: re-ranker scores (top 1, top 2, mean of top 3, gap), best dense cosine similarity, best BM25 score, how many question words appear in the top passage, and question length.

- **Document Chat / API**: below the threshold the answer is "couldn't find this" immediately, without calling the LLM (an off-topic question took 1.3 s instead of ~20 s). The confidence is shown with every answer; the skip can be turned off.
- **Research agent**: web mode `auto` (the default) searches DuckDuckGo only for sub-questions the documents can't answer.

**Training data.** For every indexed chunk the LLM writes two questions: one that may reuse the passage's words and one phrased the way an outsider would ask. Unanswerable examples come from `datasets/ml_negatives.txt`: 40 off-topic questions plus 30 near misses that use the same vocabulary as the documents ("How much does Qdrant Cloud cost?"). The file sits outside `samples/` on purpose, since anything there gets indexed.

**Results** (149 questions, 79 answerable, 5-fold cross-validation; hold-out = the 17 hand-written questions of the evaluation set, never seen in training):

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| baseline: top re-rank score alone | 0.992 | 0.940 | 0.944 |
| **logistic regression** (chosen) | **0.994** | **0.953** | **0.956** |
| gradient boosting | 0.959 | 0.940 | 0.944 |

On the hold-out, all 15 answerable questions were kept (lowest confidence 0.49) and both unanswerable ones were caught (0.15 and 0.07) at the threshold of 0.41. The threshold is the highest one that still keeps ≥ 95% of answerable questions in cross-validation, because skipping an answerable question costs more than one wasted LLM call.

Lessons from the first attempt:
- Questions copied from the passage were too easy: every model, even the baseline, scored a perfect 1.0 AUC. The cut-off landed at 0.86, and on the hold-out the model would have skipped 3 of 15 rephrased real questions. Paraphrased training questions and stronger regularization (`C=0.1`) fixed it.
- The re-ranker score alone is already a strong signal. The other features add a small, consistent gain in accuracy and F1, and make the model more robust when the re-ranker is unsure.

Fine-tuning the embedding model was considered and skipped: Stage 4 showed retrieval already finds the right chunk in the top 4 every time, and fine-tuning would need PyTorch (~200 MB+) and a GPU.

## Stage 6 — API, web UI & deployment

```bash
uv run uvicorn research_analyst.api:app --port 8000
# web UI: http://localhost:8000      interactive API docs: http://localhost:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | models in use, confidence model, number of indexed chunks |
| `GET` / `POST /api/documents`, `DELETE /api/documents/{name}` | list, upload (pdf/md/txt) and remove documents |
| `POST /api/ask` | cited answer + confidence + sources (`mode`, `rerank`, `skip_low_confidence` optional) |
| `POST /api/ask/stream` | the same as server-sent events: `meta` (confidence, sources), `token`…, `done` |
| `POST /api/research` → `POST /api/research/{id}/resume` | research agent: returns the plan for approval, then the report |
| `GET /api/research/{id}` | state of a run (runs are checkpointed in SQLite) |

The web UI (`web/index.html`) is one static page with no build step and no external scripts: document upload and removal, streaming chat with confidence and sources, and the research agent with plan review. Like the Streamlit app, the API opens `data/qdrant` in local mode, so run one of the two at a time.

**Deploying to Hugging Face Spaces (free CPU tier).** The `Dockerfile` runs the API + web UI without Ollama: answers come from Groq's free tier (`llama-3.3-70b-versatile`) and embeddings run in-process with fastembed (`nomic-ai/nomic-embed-text-v1.5-Q`, the ONNX build of the same model used locally). Models are downloaded while the image builds, and the sample documents are indexed on first start.

```bash
# needs a Hugging Face token with write access and a free Groq key (https://console.groq.com/keys)
set HF_TOKEN=hf_...
set GROQ_API_KEY=gsk_...
uv run python scripts/deploy_space.py --space <your-username>/research-analyst
```

The script creates the Space, stores the Groq key as a Space secret and uploads only the source code (~100 KB); the image is built on Hugging Face's servers. A Space's disk is temporary, so uploaded documents disappear when it restarts.

**CI**: `.github/workflows/ci.yml` runs the test suite on every push. Tests that need Ollama skip themselves; the API tests use fake embeddings and a fake LLM.

### Troubleshooting

- **`llama-server process has terminated: exit status 0xc0000005`**: Ollama tried to use an old/unsupported GPU (e.g. GeForce GT 730 via Vulkan). Force CPU by setting the user environment variables `GGML_VK_VISIBLE_DEVICES=-1` and `CUDA_VISIBLE_DEVICES=-1`, then restart Ollama.
- **`[WinError 10054] An existing connection was forcibly closed` / Ollama keeps restarting**: a VPN that hooks Windows networking can crash Ollama. Astrill installs a Winsock LSP (`ASProxy64.dll`, see `netsh winsock show catalog`) that every process loads, in any VPN mode, and Ollama dies inside `WSASend` (exit `0xc0000005`, logged in `%LOCALAPPDATA%\Ollama\server.log`). The tray app restarts the server within seconds and all LLM/embedding calls here retry, so runs survive; streaming chat only retries before the first token. To remove the cause, exclude Ollama from the VPN or remove the LSP (`netsh winsock reset`, which affects the VPN).
- **Ollama stops answering but its process is still running**: occasionally the crashed server doesn't exit, so the tray app never restarts it (`server.log` ends with a register dump). Every request here times out (`OLLAMA_TIMEOUT`, 300 s; embeddings 60 s) and is retried, but Ollama itself only recovers when you quit and restart the tray app.
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
web/index.html          no-build web UI served by the API
Dockerfile              API + web UI image (Hugging Face Spaces)
.github/workflows/      CI: tests on every push
src/research_analyst/   application package
  config.py             settings from .env
  llm.py                provider-agnostic LLM factory (+ retries, timeouts)
  embeddings.py         Ollama or fastembed embeddings with nomic task prefixes
  vectorstore.py        Qdrant local-mode store (dense + BM25 vectors)
  retrieval.py          dense / sparse / hybrid search + re-ranking
  ingest.py             load → section-aware split → index
  rag.py                retrieve → answer with citations
  agent.py              LangGraph research agent
  web.py                DuckDuckGo search (no API key)
  evaluation.py         retrieval + answer metrics
  judge.py              YES/NO LLM judge (evaluation + agent fact-check)
  tracing.py            JSONL tracer (+ optional Langfuse)
  confidence.py         answerability model: features, training, inference
  ml_data.py            training-set generation for the confidence model
  text.py               shared text helpers
  api.py                FastAPI backend
  ui.py                 shared Streamlit helpers
scripts/                entry points: hello_llm, ingest, ask, research, evaluate,
                        make_ml_dataset, train_confidence, warmup, deploy_space
samples/                sample documents + labelled evaluation set (indexed by the demo)
datasets/               confidence-model negatives (never indexed)
tests/                  pytest suite (Ollama tests auto-skip if models are missing)
data/                   local documents + vector DB (git-ignored)
notebooks/              experiments
```
