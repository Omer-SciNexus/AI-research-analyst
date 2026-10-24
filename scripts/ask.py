"""Ask a question about the indexed documents from the command line.

Usage:
    uv run python scripts/ask.py "What is hybrid search?"
"""

import sys
import time

from research_analyst.rag import retrieve, source_label, stream_answer
from research_analyst.vectorstore import get_vectorstore


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    store = get_vectorstore()

    start = time.perf_counter()
    docs = retrieve(question, store)
    print(f"Question: {question}\n")
    for chunk in stream_answer(question, docs):
        print(chunk, end="", flush=True)

    print("\n\nSources:")
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] {source_label(doc)}")
    print(f"\n[{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
