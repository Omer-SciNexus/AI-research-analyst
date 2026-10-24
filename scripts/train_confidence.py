"""Train the confidence (answerability) model on data/ml/answerability.jsonl (Stage 5).

5-fold cross-validation compares logistic regression, gradient boosting and a
baseline (the raw top re-rank score); the best is refit on all data. The
hand-written questions in samples/eval_set.json, never seen in training, serve as
a final test.

Usage:
    uv run python scripts/train_confidence.py
Output: data/models/confidence.joblib (+ .json report)
"""

import sys

import numpy as np

from research_analyst.confidence import extract_features, load_model, train
from research_analyst.config import settings
from research_analyst.evaluation import build_eval_index, load_eval_set
from research_analyst.ml_data import load_jsonl


def holdout_test(model) -> dict:
    documents, cases = load_eval_set()
    store = build_eval_index(documents)
    rows = []
    for case in cases:
        p = model.predict(extract_features(case.question, store)[0])
        rows.append({"question": case.question, "answerable": case.answerable, "confidence": round(p, 3)})
    y = np.array([r["answerable"] for r in rows])
    pred = np.array([r["confidence"] >= model.threshold for r in rows])
    return {
        "questions": len(rows),
        "accuracy": round(float((pred == y).mean()), 3),
        "answerable_kept": round(float(pred[y].mean()), 3),
        "unanswerable_caught": round(float((~pred[~y]).mean()), 3) if (~y).any() else None,
        "rows": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load_jsonl(settings.ml_dir / "answerability.jsonl")
    model = train([r["features"] for r in data], [r["label"] for r in data])
    model.report["holdout_eval_set"] = holdout_test(model)
    path = model.save()
    load_model.cache_clear()

    report = model.report
    print(f"Samples: {report['samples']} ({report['answerable']} answerable), {report['cv_folds']}-fold CV\n")
    print(f"  {'model':<24}{'roc_auc':>9}{'accuracy':>10}{'f1':>7}{'precision':>11}{'recall':>8}")
    for name, m in report["models"].items():
        mark = "  <- chosen" if name == report["chosen"] else ""
        print(f"  {name:<24}{m['roc_auc']:>9}{m['accuracy']:>10}{m['f1']:>7}{m['precision']:>11}{m['recall']:>8}{mark}")
    print(f"\nThreshold: {report['threshold']} (keeps >= 95% of answerable questions)")
    print("Feature weights:", ", ".join(f"{k}={v}" for k, v in report["importance"].items()))
    h = report["holdout_eval_set"]
    print(
        f"\nHold-out (samples/eval_set.json, {h['questions']} hand-written questions): accuracy {h['accuracy']}, "
        f"answerable kept {h['answerable_kept']}, unanswerable caught {h['unanswerable_caught']}"
    )
    for row in h["rows"]:
        print(f"  {row['confidence']:.2f} {'answerable  ' if row['answerable'] else 'UNANSWERABLE'} {row['question'][:70]}")
    print(f"\n[saved to {path}]")


if __name__ == "__main__":
    main()
