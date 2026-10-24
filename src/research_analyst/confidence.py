"""Stage 5: answerability ("confidence") model.

Predicts P(the indexed documents can answer this question) from retrieval signals
alone, before any LLM call - a few milliseconds instead of ~7 s of CPU generation.
Low-confidence questions can be answered "not found" immediately (Document Chat)
or sent to web search (Research Agent).

Features come from one hybrid + re-ranked search plus a dense and a BM25 probe:

    rerank_top1 / rerank_top2 / rerank_mean3 / rerank_gap   cross-encoder relevance
    dense_top1                                               cosine similarity of the best chunk
    bm25_top1                                                BM25 score of the best keyword match
    query_coverage                                           question words found in the top passage
    query_words                                              question length (content words)

Trained by scripts/train_confidence.py on questions the LLM wrote from the indexed
chunks (answerable) and off-topic / near-miss questions (unanswerable).
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_analyst.config import settings
from research_analyst.retrieval import search
from research_analyst.text import content_words, coverage
from research_analyst.vectorstore import with_mode

FEATURES = [
    "rerank_top1",
    "rerank_top2",
    "rerank_mean3",
    "rerank_gap",
    "dense_top1",
    "bm25_top1",
    "query_coverage",
    "query_words",
]


# ── Features ─────────────────────────────────────────────────────────────────


def extract_features(question: str, store: QdrantVectorStore) -> tuple[dict[str, float], list[Document]]:
    """Retrieval signals for `question`, plus the re-ranked documents (reusable for answering)."""
    docs = search(question, store, mode="hybrid", use_rerank=True, k=settings.top_k)
    scores = [d.metadata.get("rerank_score", 0.0) for d in docs] + [0.0] * 3
    dense = with_mode(store, "dense").similarity_search_with_score(question, k=1)
    sparse = with_mode(store, "sparse").similarity_search_with_score(question, k=1)
    words = content_words(question)
    features = {
        "rerank_top1": scores[0],
        "rerank_top2": scores[1],
        "rerank_mean3": float(np.mean(scores[:3])),
        "rerank_gap": scores[0] - scores[1],
        "dense_top1": dense[0][1] if dense else 0.0,
        "bm25_top1": sparse[0][1] if sparse else 0.0,
        "query_coverage": (coverage(words, docs[0].page_content) or 0.0) if docs else 0.0,
        "query_words": float(len(words)),
    }
    return features, docs


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[row[name] for name in FEATURES] for row in rows], dtype=float)


# ── Training ─────────────────────────────────────────────────────────────────


CANDIDATES = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=0),
}


def _scores(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 3),
        "accuracy": round(accuracy_score(y, pred), 3),
        "f1": round(f1_score(y, pred), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred), 3),
    }


def pick_threshold(y: np.ndarray, proba: np.ndarray, min_recall: float = 0.95) -> float:
    """Highest threshold that still keeps `min_recall` of answerable questions.

    Skipping the LLM for an answerable question is the costly mistake, so recall on
    answerable questions is protected first; within that, the threshold is as high
    as possible to catch more unanswerable ones.
    """
    for threshold in np.round(np.arange(0.95, 0.0, -0.01), 2):
        if recall_score(y, (proba >= threshold).astype(int)) >= min_recall:
            return float(threshold)
    return 0.05


@dataclass
class ConfidenceModel:
    estimator: object
    name: str
    threshold: float
    report: dict = field(default_factory=dict)

    def predict(self, features: dict[str, float]) -> float:
        return float(self.estimator.predict_proba(to_matrix([features]))[0, 1])

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or settings.confidence_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        return path


def train(rows: list[dict], labels: list[int], folds: int = 5) -> ConfidenceModel:
    """Cross-validate every candidate (and a single-feature baseline), keep the best by ROC-AUC."""
    X, y = to_matrix(rows), np.array(labels)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    report: dict = {"samples": len(y), "answerable": int(y.sum()), "features": FEATURES, "cv_folds": folds, "models": {}}

    # Baseline: the raw top re-rank score, thresholded the same way
    baseline = X[:, FEATURES.index("rerank_top1")]
    report["models"]["baseline_rerank_top1"] = _scores(y, baseline, pick_threshold(y, baseline))

    best_name, best_auc, best_oof = None, -1.0, None
    for name, estimator in CANDIDATES.items():
        oof = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")[:, 1]
        threshold = pick_threshold(y, oof)
        report["models"][name] = {**_scores(y, oof, threshold), "threshold": threshold}
        if report["models"][name]["roc_auc"] > best_auc:
            best_name, best_auc, best_oof = name, report["models"][name]["roc_auc"], oof

    estimator = clone(CANDIDATES[best_name]).fit(X, y)
    threshold = pick_threshold(y, best_oof)
    report.update(chosen=best_name, threshold=threshold, importance=_importance(estimator))
    return ConfidenceModel(estimator=estimator, name=best_name, threshold=threshold, report=report)


def _importance(estimator) -> dict[str, float]:
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:  # pipeline ending in logistic regression: standardized coefficients
        values = estimator[-1].coef_[0]
    return {name: round(float(v), 3) for name, v in zip(FEATURES, values)}


# ── Inference ────────────────────────────────────────────────────────────────


@lru_cache
def load_model(path: str | None = None) -> ConfidenceModel | None:
    path = Path(path or settings.confidence_model_path)
    return joblib.load(path) if path.exists() else None


@dataclass
class Assessment:
    docs: list[Document]
    confidence: float | None  # None when no model has been trained yet
    threshold: float | None

    @property
    def answerable(self) -> bool:
        return self.confidence is None or self.confidence >= self.threshold


def assess(question: str, store: QdrantVectorStore) -> Assessment:
    features, docs = extract_features(question, store)
    model = load_model()
    if model is None:
        return Assessment(docs=docs, confidence=None, threshold=None)
    return Assessment(docs=docs, confidence=model.predict(features), threshold=model.threshold)
