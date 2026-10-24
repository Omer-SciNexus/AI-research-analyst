"""Document chat: upload documents and chat with them (Stages 1, 2 and 5).

Run:  uv run streamlit run app.py
"""

import streamlit as st

from research_analyst.confidence import load_model
from research_analyst.config import settings
from research_analyst.ingest import SUPPORTED_SUFFIXES, ingest_file
from research_analyst.rag import NOT_FOUND, prepare, source_label, stream_answer
from research_analyst.ui import get_store as store
from research_analyst.vectorstore import MODES, list_sources

st.set_page_config(page_title="Research Analyst", page_icon="📚", layout="wide")


with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader(
        "Add PDF, Markdown or text files",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index files", type="primary"):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            path = settings.docs_dir / upload.name
            path.write_bytes(upload.getbuffer())
            with st.spinner(f"Indexing {upload.name}…"):
                count = ingest_file(path, store())
            st.success(f"{upload.name}: {count} chunks")

    st.subheader("Indexed")
    sources = list_sources(store())
    if sources:
        for name, count in sorted(sources.items()):
            st.caption(f"• {name} — {count} chunks")
    else:
        st.caption("Nothing indexed yet.")

    st.subheader("🔎 Retrieval")
    mode = st.selectbox(
        "Search mode",
        list(MODES),
        index=list(MODES).index(settings.retrieval_mode),
        help="dense = meaning (embeddings) · sparse = keywords (BM25) · hybrid = both, fused with RRF",
    )
    use_rerank = st.toggle("Re-rank results", value=settings.rerank, help=f"Cross-encoder: {settings.reranker_model}")

    st.subheader("🎯 Confidence model")
    model = load_model()
    if model:
        skip_low = st.toggle(
            "Skip the LLM when documents can't answer",
            value=settings.skip_low_confidence,
            help=f"{model.name}, threshold {model.threshold}: replies 'not found' instantly instead of generating",
        )
    else:
        skip_low = False
        st.caption("Not trained yet: `uv run python scripts/train_confidence.py`")

    st.divider()
    st.caption(f"LLM: {settings.llm_provider} / {settings.llm_model}")
    st.caption(f"Embeddings: {settings.embedding_model} + {settings.sparse_model}")
    if st.button("Clear chat"):
        st.session_state.messages = []


st.title("📚 Document Chat")
st.caption("Ask questions about your documents. Answers cite the passages they use.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            score = f" · rerank {src['score']:.3f}" if src.get("score") is not None else ""
            st.markdown(f"**[{i}] {src['label']}**{score}")
            st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))


def confidence_note(turn) -> str:
    if turn.confidence is None:
        return ""
    verdict = "documents can likely answer" if turn.confidence >= turn.threshold else "documents likely can't answer"
    return f"🎯 Confidence {turn.confidence:.2f} — {verdict}" + (" · LLM skipped" if turn.skip_llm else "")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("note"):
            st.caption(message["note"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            turn = prepare(question, store(), mode=mode, use_rerank=use_rerank, skip_low_confidence=skip_low)
        if turn.skip_llm:
            reply = NOT_FOUND
            st.markdown(reply)
        else:
            reply = st.write_stream(stream_answer(question, turn.docs))
        note = confidence_note(turn)
        if note:
            st.caption(note)
        sources = [
            {"label": source_label(d), "text": d.page_content, "score": d.metadata.get("rerank_score")}
            for d in turn.docs
        ]
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "note": note, "sources": sources})
