import numpy as np
import pytest
from test_rag import needs_embeddings

from research_analyst.confidence import (
    FEATURES,
    Assessment,
    ConfidenceModel,
    extract_features,
    pick_threshold,
    train,
)
from research_analyst.ml_data import load_negatives, parse_questions


def synthetic(n: int = 120, seed: int = 0) -> tuple[list[dict], list[int]]:
    """Answerable questions get high re-rank scores, unanswerable ones low, with noise."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n):
        label = i % 2
        top = rng.normal(0.8 if label else 0.15, 0.15)
        rows.append({
            "rerank_top1": top, "rerank_top2": top * 0.7, "rerank_mean3": top * 0.6, "rerank_gap": top * 0.3,
            "dense_top1": rng.normal(0.75 if label else 0.55, 0.05), "bm25_top1": rng.normal(8 if label else 3, 2),
            "query_coverage": rng.uniform(0.5, 1) if label else rng.uniform(0, 0.4), "query_words": rng.integers(3, 9),
        })
        labels.append(label)
    return rows, labels


def test_pick_threshold_protects_answerable_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.1, 0.05, 0.6])
    threshold = pick_threshold(y, proba, min_recall=0.75)
    assert threshold == 0.7  # highest cut keeping 3 of 4 answerable
    assert pick_threshold(y, proba, min_recall=1.0) == 0.2


def test_train_learns_separable_data_and_beats_nothing(tmp_path):
    rows, labels = synthetic()
    model = train(rows, labels)
    assert model.name in {"logistic_regression", "gradient_boosting"}
    assert model.report["models"][model.name]["roc_auc"] > 0.95
    assert set(model.report["importance"]) == set(FEATURES)

    good = {**rows[1], "rerank_top1": 0.95, "query_coverage": 0.9}
    bad = {**rows[0], "rerank_top1": 0.01, "query_coverage": 0.0}
    assert model.predict(good) > model.threshold > model.predict(bad)

    path = model.save(tmp_path / "m.joblib")
    assert path.with_suffix(".json").exists()


def test_assessment_without_model_never_blocks():
    assert Assessment(docs=[], confidence=None, threshold=None).answerable
    assert not Assessment(docs=[], confidence=0.1, threshold=0.4).answerable


def test_parse_questions_strips_markers_and_noise():
    text = 'Here are questions:\n1. What chunk size is used?\n- "Why use Qdrant?"\nNot a question\n2) How?'
    assert parse_questions(text) == ["What chunk size is used?", "Why use Qdrant?"]


def test_negatives_file_has_off_topic_and_near_miss_questions():
    negatives = load_negatives()
    assert len(negatives) >= 60
    assert any("Qdrant" in q for q in negatives) and any("pizza" in q for q in negatives)


@needs_embeddings
def test_extract_features_returns_every_feature():
    from test_rag import SAMPLE
    from qdrant_client import QdrantClient

    from research_analyst.ingest import ingest_file
    from research_analyst.vectorstore import build_vectorstore

    store = build_vectorstore(QdrantClient(":memory:"), "f")
    ingest_file(SAMPLE, store)
    on_topic, docs = extract_features("What chunk size does the project use?", store)
    off_topic, _ = extract_features("How do I bake sourdough bread?", store)
    assert set(on_topic) == set(FEATURES) and docs
    assert on_topic["rerank_top1"] > off_topic["rerank_top1"]
    assert on_topic["query_coverage"] > off_topic["query_coverage"]


def test_model_type_is_saved_with_threshold(tmp_path):
    rows, labels = synthetic(60, seed=1)
    model = train(rows, labels, folds=3)
    assert isinstance(model, ConfidenceModel) and 0 < model.threshold < 1
    with pytest.raises(KeyError):
        model.predict({"rerank_top1": 0.5})  # every feature is required
