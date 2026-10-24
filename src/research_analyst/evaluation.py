"""Stage 4: offline evaluation against a labelled question set (samples/eval_set.json).

Retrieval (no LLM, ~seconds): for every search setup, where does the first chunk from the
labelled section rank?  hit@1, hit@k and MRR (mean reciprocal rank).

Answers (needs the LLM, minutes on CPU): keyword recall, citation rate, refusal accuracy on
unanswerable questions, and faithfulness judged by the LLM against the retrieved passages.

The documents are indexed into a fresh in-memory collection, so results are reproducible and
the app's own index is untouched.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.ingest import ingest_file
from research_analyst.judge import is_supported
from research_analyst.rag import NOT_FOUND, format_context, generate_answer
from research_analyst.retrieval import search
from research_analyst.text import content_words, coverage
from research_analyst.vectorstore import build_vectorstore

SETUPS = {
    "dense": {"mode": "dense", "use_rerank": False},
    "sparse (BM25)": {"mode": "sparse", "use_rerank": False},
    "hybrid": {"mode": "hybrid", "use_rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "use_rerank": True},
}
CITATION = re.compile(r"\[\d+\]")


@dataclass
class EvalCase:
    question: str
    source: str | None
    section: str | None
    keywords: list[str]

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_eval_set(path: Path | None = None) -> tuple[list[Path], list[EvalCase]]:
    data = json.loads(Path(path or settings.eval_set).read_text(encoding="utf-8"))
    documents = [PROJECT_ROOT / p for p in data["documents"]]
    return documents, [EvalCase(**case) for case in data["cases"]]


def build_eval_index(documents: list[Path]) -> QdrantVectorStore:
    store = build_vectorstore(QdrantClient(":memory:"), "eval")
    for path in documents:
        ingest_file(path, store)
    return store


def is_relevant(doc: Document, case: EvalCase) -> bool:
    if doc.metadata.get("source") != case.source:
        return False
    return case.section is None or doc.metadata.get("section", "").split(" > ")[-1] == case.section


def first_relevant_rank(docs: list[Document], case: EvalCase) -> int | None:
    return next((rank for rank, doc in enumerate(docs, start=1) if is_relevant(doc, case)), None)


def evaluate_retrieval(store: QdrantVectorStore, cases: list[EvalCase], k: int | None = None) -> dict:
    k = k or settings.top_k
    answerable = [c for c in cases if c.answerable]
    results = {}
    for name, options in SETUPS.items():
        ranks, latencies = [], []
        for case in answerable:
            start = time.perf_counter()
            docs = search(case.question, store, k=k, **options)
            latencies.append(time.perf_counter() - start)
            ranks.append(first_relevant_rank(docs, case))
        results[name] = {
            "hit@1": round(mean(float(r == 1) for r in ranks), 3),
            f"hit@{k}": round(mean(float(r is not None) for r in ranks), 3),
            "mrr": round(mean(1 / r if r else 0.0 for r in ranks), 3),
            "latency_ms": round(1000 * mean(latencies), 1),
            "ranks": {c.question: r for c, r in zip(answerable, ranks)},
        }
    return results


def judge_faithfulness(answer: str, docs: list[Document]) -> bool | None:
    return is_supported(answer, format_context(docs))


def lexical_grounding(answer: str, docs: list[Document]) -> float | None:
    """Share of the answer's content words that appear in the retrieved passages.

    A cheap, judge-free proxy for faithfulness: invented facts bring in words the
    passages don't contain.
    """
    share = coverage(content_words(answer), " ".join(d.page_content for d in docs))
    return None if share is None else round(share, 3)


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    return NOT_FOUND.lower().rstrip(".") in text or "could not find" in text or "couldn't find" in text


def evaluate_answers(store: QdrantVectorStore, cases: list[EvalCase], setup: str = "hybrid + rerank") -> dict:
    rows = []
    for case in cases:
        start = time.perf_counter()
        docs = search(case.question, store, **SETUPS[setup])
        answer = generate_answer(case.question, docs)
        row = {"question": case.question, "answer": answer, "latency_s": round(time.perf_counter() - start, 1)}
        if case.answerable:
            lowered = answer.lower()
            row["keyword_recall"] = mean(kw.lower() in lowered for kw in case.keywords) if case.keywords else None
            row["cited"] = bool(CITATION.search(answer))
            refused = is_refusal(answer)
            row["grounding"] = None if refused else lexical_grounding(answer, docs)
            row["faithful"] = None if refused else judge_faithfulness(answer, docs)
        else:
            row["refused"] = is_refusal(answer)
        rows.append(row)

    def avg(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(mean(values), 3) if values else None

    return {
        "setup": setup,
        "keyword_recall": avg("keyword_recall"),
        "citation_rate": avg("cited"),
        "grounding": avg("grounding"),
        "faithfulness": avg("faithful"),
        "refusal_accuracy": avg("refused"),
        "latency_s": avg("latency_s"),
        "rows": rows,
    }


def save_results(results: dict) -> Path:
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    path = settings.eval_dir / f"results-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def latest_results() -> tuple[Path, dict] | None:
    paths = sorted(settings.eval_dir.glob("results-*.json"))
    return (paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))) if paths else None


def run_evaluation(with_answers: bool = False) -> dict:
    documents, cases = load_eval_set()
    store = build_eval_index(documents)
    results = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {**asdict(EvalConfig()), "cases": len(cases)},
        "retrieval": evaluate_retrieval(store, cases),
    }
    if with_answers:
        results["answers"] = evaluate_answers(store, cases)
    return results


@dataclass
class EvalConfig:
    """Settings that affect the scores, saved with every result for comparison."""

    embedding_model: str = settings.embedding_model
    llm_model: str = settings.llm_model
    judge_model: str = f"{settings.judge_provider}/{settings.judge_model}"
    reranker_model: str = settings.reranker_model
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
    top_k: int = settings.top_k
    fetch_k: int = settings.fetch_k
