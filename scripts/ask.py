"""Ask a question about the indexed documents from the command line.

Usage:
    uv run python scripts/ask.py "What is hybrid search?"
    uv run python scripts/ask.py --mode dense --no-rerank "What is hybrid search?"
"""

import argparse
import time

from research_analyst.config import settings
from research_analyst.rag import retrieve, source_label, stream_answer
from research_analyst.vectorstore import MODES, get_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", choices=list(MODES), default=settings.retrieval_mode)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    question = " ".join(args.question) or "What is retrieval-augmented generation?"
    store = get_vectorstore()

    start = time.perf_counter()
    docs = retrieve(question, store, mode=args.mode, use_rerank=not args.no_rerank)
    print(f"Question: {question}\n[retrieval: {args.mode}{'' if args.no_rerank else ' + rerank'}]\n")
    for chunk in stream_answer(question, docs):
        print(chunk, end="", flush=True)

    print("\n\nSources:")
    for i, doc in enumerate(docs, start=1):
        score = doc.metadata.get("rerank_score")
        print(f"  [{i}] {source_label(doc)}" + (f"  (rerank {score:.3f})" if score is not None else ""))
    print(f"\n[{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
