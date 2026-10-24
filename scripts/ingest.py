"""Index documents (PDF / Markdown / text) into the local vector store.

Usage:
    uv run python scripts/ingest.py                 # everything in data/docs
    uv run python scripts/ingest.py samples         # a folder
    uv run python scripts/ingest.py paper.pdf       # a single file

Stop the Streamlit app first: Qdrant local mode allows one process at a time.
"""

import sys
import time
from pathlib import Path

from research_analyst.config import settings
from research_analyst.ingest import find_files, ingest_file
from research_analyst.vectorstore import get_vectorstore, list_sources


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.docs_dir
    files = find_files(target)
    if not files:
        print(f"No .pdf/.md/.txt files found in {target}")
        return

    store = get_vectorstore()
    for file in files:
        start = time.perf_counter()
        count = ingest_file(file, store)
        print(f"  {file.name}: {count} chunks ({time.perf_counter() - start:.1f}s)")

    sources = list_sources(store)
    print(f"\nIndex now holds {sum(sources.values())} chunks from {len(sources)} files.")


if __name__ == "__main__":
    main()
