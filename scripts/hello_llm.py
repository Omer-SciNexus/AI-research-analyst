"""Stage 0 smoke test: ask the configured LLM a question and stream the answer.

Usage:
    uv run python scripts/hello_llm.py
    uv run python scripts/hello_llm.py "What is retrieval-augmented generation?"
"""

import sys
import time

from research_analyst.config import settings
from research_analyst.llm import get_llm


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Explain RAG (retrieval-augmented generation) in 3 short bullet points."
    llm = get_llm()

    print(f"Provider: {settings.llm_provider} | Model: {settings.llm_model}")
    print(f"Question: {question}\n")

    start = time.perf_counter()
    first_token_at = None
    for chunk in llm.stream(question):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        print(chunk.content, end="", flush=True)
    end = time.perf_counter()

    print(f"\n\n[first token: {first_token_at - start:.1f}s | total: {end - start:.1f}s]")


if __name__ == "__main__":
    main()
