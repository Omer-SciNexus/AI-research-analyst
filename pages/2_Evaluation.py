"""Stage 4-5 UI: evaluation scores, the confidence model and local traces."""

import json

import pandas as pd
import streamlit as st

from research_analyst.config import settings
from research_analyst.evaluation import latest_results, run_evaluation, save_results
from research_analyst.tracing import load_records

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation & Traces")

score = st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")

tab_eval, tab_model, tab_traces = st.tabs(["Evaluation", "Confidence model", "Traces"])

with tab_model:
    report_path = settings.confidence_model_path.with_suffix(".json")
    if not report_path.exists():
        st.info(
            "Not trained yet. Run `uv run python scripts/make_ml_dataset.py` "
            "then `uv run python scripts/train_confidence.py`."
        )
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Predicts whether the documents can answer a question from retrieval signals, before any LLM call. "
            f"{report['samples']} training questions ({report['answerable']} answerable), "
            f"{report['cv_folds']}-fold cross-validation. Chosen: **{report['chosen']}**, threshold **{report['threshold']}** "
            "(keeps ≥ 95% of answerable questions)."
        )
        models = pd.DataFrame(report["models"]).T.drop(columns="threshold", errors="ignore")
        st.dataframe(models, column_config={c: score for c in models.columns}, width="stretch")

        left, right = st.columns(2)
        with left:
            st.markdown("**Feature weights**")
            weights = pd.DataFrame({"weight": report["importance"]}).sort_values("weight", key=abs, ascending=False)
            st.dataframe(weights, width="stretch")
        with right:
            holdout = report.get("holdout_eval_set")
            if holdout:
                st.markdown(f"**Hold-out: {holdout['questions']} hand-written questions**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", holdout["accuracy"])
                c2.metric("Answerable kept", holdout["answerable_kept"])
                c3.metric("Unanswerable caught", "—" if holdout["unanswerable_caught"] is None else holdout["unanswerable_caught"])
                st.dataframe(pd.DataFrame(holdout["rows"]), width="stretch", hide_index=True)

with tab_eval:
    st.caption(
        "Labelled questions in `samples/eval_set.json`, indexed into a fresh in-memory collection. "
        "Answer metrics come from `uv run python scripts/evaluate.py --answers` (needs the LLM)."
    )
    if st.button("▶️ Run retrieval evaluation (~30 s)", type="primary"):
        with st.spinner("Indexing the eval documents and searching with every setup…"):
            save_results(run_evaluation(with_answers=False))
        st.rerun()

    latest = latest_results()
    if not latest:
        st.info("No results yet. Run the evaluation above.")
    else:
        path, results = latest
        config = results["config"]
        st.caption(
            f"Latest: `{path.name}` · {config['cases']} questions · top-{config['top_k']} · "
            f"{config['embedding_model']} · reranker {config['reranker_model']}"
        )

        st.subheader("Retrieval")
        retrieval = results["retrieval"]
        table = pd.DataFrame(
            {setup: {m: v for m, v in scores.items() if m != "ranks"} for setup, scores in retrieval.items()}
        ).T
        metric_cols = [c for c in table.columns if c != "latency_ms"]
        st.dataframe(
            table,
            column_config={**{c: score for c in metric_cols}, "latency_ms": st.column_config.NumberColumn("latency (ms)", format="%.0f")},
            width="stretch",
        )
        st.caption("hit@1: right chunk ranked first · hit@k: right chunk anywhere in the top k · MRR: mean of 1/rank")

        with st.expander("Rank of the correct chunk, per question (blank = not in top k)"):
            ranks = pd.DataFrame({setup: scores["ranks"] for setup, scores in retrieval.items()})
            st.dataframe(ranks, width="stretch")

        if answers := results.get("answers"):
            st.subheader(f"Answers ({answers['setup']})")
            metrics = [
                ("Keyword recall", "keyword_recall", "expected facts found in the answer"),
                ("Citation rate", "citation_rate", "answers citing at least one passage"),
                ("Grounding", "grounding", "answer content words found in the passages (no LLM)"),
                ("Faithfulness", "faithfulness", f"LLM judge ({config.get('judge_model', '?')}); a 3B judge is unreliable"),
                ("Refusal accuracy", "refusal_accuracy", "unanswerable questions correctly refused"),
                ("Latency (s)", "latency_s", "retrieval + answer per question"),
            ]
            for col, (label, key, help_text) in zip(st.columns(len(metrics)), metrics):
                col.metric(label, "—" if answers.get(key) is None else answers[key], help=help_text)
            st.dataframe(pd.DataFrame(answers["rows"]), width="stretch", hide_index=True)

with tab_traces:
    records = load_records()
    if not records:
        st.info("No traces yet. Ask something in Document Chat or run the Research Agent.")
    else:
        traces = [r for r in records if r["kind"] == "trace"]
        llm_calls = pd.DataFrame([r for r in records if r["kind"] == "llm"])

        rows = []
        for trace in traces:
            calls = llm_calls[llm_calls["trace_id"] == trace["trace_id"]] if not llm_calls.empty else llm_calls
            rows.append({
                "time": pd.to_datetime(trace["start"], unit="s"),
                "run": trace["name"],
                "duration (s)": trace["duration_s"],
                "LLM calls": len(calls),
                "tokens in": int(calls["input_tokens"].fillna(0).sum()) if len(calls) else 0,
                "tokens out": int(calls["output_tokens"].fillna(0).sum()) if len(calls) else 0,
                "error": trace.get("error", ""),
                "trace_id": trace["trace_id"],
            })
        overview = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("Runs (7 days)", len(overview))
        c2.metric("LLM calls", len(llm_calls))
        if not llm_calls.empty and llm_calls["output_tokens"].notna().any():
            speed = llm_calls["output_tokens"].sum() / llm_calls["duration_s"].sum()
            c3.metric("Avg generation speed", f"{speed:.1f} tok/s")

        st.dataframe(overview.drop(columns="trace_id"), width="stretch", hide_index=True)

        choice = st.selectbox(
            "Inspect a run",
            overview.index,
            format_func=lambda i: f"{overview.at[i, 'time']:%H:%M:%S} · {overview.at[i, 'run']} · {overview.at[i, 'duration (s)']:.1f}s",
        )
        if choice is not None and not llm_calls.empty:
            calls = llm_calls[llm_calls["trace_id"] == overview.at[choice, "trace_id"]].sort_values("start")
            calls = calls.assign(started=pd.to_datetime(calls["start"], unit="s"))
            st.dataframe(
                calls[["started", "name", "model", "duration_s", "input_tokens", "output_tokens", "input_chars", "output_chars"]],
                width="stretch",
                hide_index=True,
            )
