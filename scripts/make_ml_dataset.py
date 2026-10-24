"""Build the confidence-model training set (Stage 5).

Indexes samples/ and data/docs into a fresh in-memory collection, asks the LLM for
questions about every chunk, adds samples/ml_negatives.txt, and computes retrieval
features for each question. ~10 minutes with llama3.2:3b on CPU.

Usage:
    uv run python scripts/make_ml_dataset.py
Output: data/ml/answerability.jsonl
"""

import sys
import time

from qdrant_client import QdrantClient

from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.ingest import find_files, ingest_file
from research_analyst.ml_data import featurize, load_negatives, positive_questions, save_jsonl
from research_analyst.vectorstore import build_vectorstore


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    start = time.perf_counter()
    store = build_vectorstore(QdrantClient(":memory:"), "ml")
    files = find_files(PROJECT_ROOT / "samples") + find_files(settings.docs_dir)
    for file in files:
        ingest_file(file, store)
    print(f"Indexed {len(files)} files.")

    positives = []
    for row in positive_questions(store):
        positives.append(row)
        print(f"  [{time.perf_counter() - start:5.0f}s] + {row['question']}")
    negatives = [{"question": q, "label": 0, "origin": "ml_negatives.txt"} for q in load_negatives()]
    print(f"\n{len(positives)} answerable + {len(negatives)} unanswerable questions; computing features…")

    rows = featurize(positives + negatives, store)
    path = settings.ml_dir / "answerability.jsonl"
    save_jsonl(rows, path)
    print(f"[saved {len(rows)} rows to {path} in {time.perf_counter() - start:.0f}s]")


if __name__ == "__main__":
    main()
