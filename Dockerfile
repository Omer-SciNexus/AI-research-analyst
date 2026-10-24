# Research Analyst - API + web UI in one container (used for Hugging Face Spaces).
# No Ollama inside: the LLM is a hosted free tier (Groq by default, key passed as a
# secret) and embeddings run in-process with fastembed (ONNX, CPU).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

# Hugging Face Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/home/user/app/.venv/bin:$PATH" \
    LLM_PROVIDER=groq \
    LLM_MODEL=llama-3.3-70b-versatile \
    EMBEDDING_PROVIDER=fastembed \
    EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5-Q \
    INDEX_SAMPLES_ON_START=true

# Dependencies first, so code changes don't reinstall them
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra groq --no-install-project

COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user samples ./samples
COPY --chown=user scripts/warmup.py ./scripts/warmup.py
RUN uv sync --frozen --no-dev --extra groq && python scripts/warmup.py

EXPOSE 7860
CMD ["uvicorn", "research_analyst.api:app", "--host", "0.0.0.0", "--port", "7860"]
