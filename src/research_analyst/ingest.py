"""Load documents, split them into chunks and index them in the vector store."""

import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

from research_analyst.config import settings
from research_analyst.vectorstore import delete_source

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_file(path: Path) -> list[Document]:
    """One Document per PDF page (so citations can point to a page), or one per text file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [
            Document(page_content=text, metadata={"source": path.name, "page": number})
            for number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Document(page_content=text, metadata={"source": path.name})] if text else []


def split_markdown_sections(doc: Document) -> list[Document]:
    """Split Markdown on headings so chunks never straddle two sections."""
    splitter = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS, strip_headers=False)
    sections = []
    for part in splitter.split_text(doc.page_content):
        metadata = dict(doc.metadata)
        if path := " > ".join(part.metadata[key] for _, key in MARKDOWN_HEADERS if key in part.metadata):
            metadata["section"] = path
        sections.append(Document(page_content=part.page_content, metadata=metadata))
    return sections


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for doc in docs:
        is_markdown = doc.metadata.get("source", "").lower().endswith(".md")
        for chunk in splitter.split_documents(split_markdown_sections(doc) if is_markdown else [doc]):
            # Continuation chunks lose their heading; prefix the section path so they stay findable.
            section = chunk.metadata.get("section")
            if section and not chunk.page_content.lstrip().startswith("#"):
                chunk.page_content = f"[{section}]\n{chunk.page_content}"
            chunks.append(chunk)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


def chunk_ids(source: str, count: int) -> list[str]:
    """Deterministic ids, so re-indexing a file overwrites instead of duplicating."""
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{i}")) for i in range(count)]


def ingest_file(path: Path, store: QdrantVectorStore) -> int:
    path = Path(path)
    chunks = split_documents(load_file(path))
    delete_source(store, path.name)  # drop stale chunks if the file changed
    if chunks:
        store.add_documents(chunks, ids=chunk_ids(path.name, len(chunks)))
    return len(chunks)


def find_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def ingest_path(path: Path, store: QdrantVectorStore) -> dict[str, int]:
    return {file.name: ingest_file(file, store) for file in find_files(path)}
