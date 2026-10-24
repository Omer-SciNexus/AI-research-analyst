from pathlib import Path

from langchain_core.documents import Document

from research_analyst.ingest import chunk_ids, find_files, load_file, split_documents
from research_analyst.rag import format_context, source_label

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_load_markdown_sets_source():
    docs = load_file(SAMPLES / "rag_handbook.md")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "rag_handbook.md"
    assert "Qdrant" in docs[0].page_content


def test_split_keeps_metadata_and_numbers_chunks():
    chunks = split_documents(load_file(SAMPLES / "rag_handbook.md"))
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "rag_handbook.md" for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic_and_unique():
    assert chunk_ids("a.pdf", 3) == chunk_ids("a.pdf", 3)
    assert len(set(chunk_ids("a.pdf", 3) + chunk_ids("b.pdf", 3))) == 6


def test_find_files_filters_by_suffix(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.md").write_text("x")
    (tmp_path / "c.png").write_bytes(b"")
    assert [p.name for p in find_files(tmp_path)] == ["a.pdf", "b.md"]


def test_format_context_numbers_passages_with_page_labels():
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="beta", metadata={"source": "b.md"}),
    ]
    assert source_label(docs[0]) == "a.pdf, p.3"
    assert format_context(docs) == "[1] (a.pdf, p.3)\nalpha\n\n[2] (b.md)\nbeta"
