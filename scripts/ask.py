"""Ask a question about the indexed documents from the command line.

Usage:
    uv run python scripts/ask.py "What is hybrid search?"
    uv run python scripts/ask.py --mode dense --no-rerank "What is hybrid search?"
    uv run python scripts/ask.py --always-answer "..."   # call the LLM even if confidence is low
"""

import argparse
import sys
import time

from research_analyst.config import settings
from research_analyst.rag import NOT_FOUND, prepare, source_label, stream_answer
from research_analyst.vectorstore import MODES, get_vectorstore


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", choices=list(MODES), default=settings.retrieval_mode)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--always-answer", action="store_true", help="don't skip the LLM on low confidence")
    args = parser.parse_args()

    question = " ".join(args.question) or "What is retrieval-augmented generation?"
    store = get_vectorstore()

    start = time.perf_counter()
    turn = prepare(question, store, mode=args.mode, use_rerank=not args.no_rerank, skip_low_confidence=not args.always_answer)
    print(f"Question: {question}\n[retrieval: {args.mode}{'' if args.no_rerank else ' + rerank'}]")
    if turn.confidence is not None:
        print(f"[confidence: {turn.confidence:.2f}, threshold {turn.threshold}{' -> LLM skipped' if turn.skip_llm else ''}]")
    print()
    for chunk in [NOT_FOUND] if turn.skip_llm else stream_answer(question, turn.docs):
        print(chunk, end="", flush=True)

    print("\n\nSources:")
    for i, doc in enumerate(turn.docs, start=1):
        score = doc.metadata.get("rerank_score")
        print(f"  [{i}] {source_label(doc)}" + (f"  (rerank {score:.3f})" if score is not None else ""))
    print(f"\n[{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
