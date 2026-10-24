"""Stage 1 UI: upload documents and chat with them.

Run:  uv run streamlit run app.py
"""

import streamlit as st

from research_analyst.config import settings
from research_analyst.ingest import SUPPORTED_SUFFIXES, ingest_file
from research_analyst.rag import retrieve, source_label, stream_answer
from research_analyst.vectorstore import get_vectorstore, list_sources

st.set_page_config(page_title="Research Analyst", page_icon="📚", layout="wide")


@st.cache_resource
def store():
    # Cached so reruns reuse one client (Qdrant local mode locks its folder).
    return get_vectorstore()


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

    st.divider()
    st.caption(f"LLM: {settings.llm_provider} / {settings.llm_model}")
    st.caption(f"Embeddings: {settings.embedding_model}")
    if st.button("Clear chat"):
        st.session_state.messages = []


st.title("📚 Research Analyst")
st.caption("Ask questions about your documents. Answers cite the passages they use.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {src['label']}**")
            st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            docs = retrieve(question, store())
        reply = st.write_stream(stream_answer(question, docs))
        sources = [{"label": source_label(d), "text": d.page_content} for d in docs]
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
