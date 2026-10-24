"""Index documents (PDF / Markdown / text) into the local vector store.

Usage:
    uv run python scripts/ingest.py                 # everything in data/docs
    uv run python scripts/ingest.py samples         # a folder
    uv run python scripts/ingest.py paper.pdf       # a single file
    uv run python scripts/ingest.py --reset samples # wipe the index first

Stop the Streamlit app first: Qdrant local mode allows one process at a time.
"""

import argparse
import time
from pathlib import Path

from research_analyst.config import settings
from research_analyst.ingest import find_files, ingest_file
from research_analyst.vectorstore import get_vectorstore, list_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, default=[settings.docs_dir])
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    files = [f for path in args.paths for f in find_files(path)]
    if not files and not args.reset:
        print(f"No .pdf/.md/.txt files found in {', '.join(map(str, args.paths))}")
        return

    store = get_vectorstore(reset=args.reset)
    if args.reset:
        print(f"Collection '{settings.collection_name}' reset.")
    for file in files:
        start = time.perf_counter()
        count = ingest_file(file, store)
        print(f"  {file.name}: {count} chunks ({time.perf_counter() - start:.1f}s)")

    sources = list_sources(store)
    print(f"\nIndex now holds {sum(sources.values())} chunks from {len(sources)} files.")


if __name__ == "__main__":
    main()
