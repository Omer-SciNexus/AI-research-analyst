# RAG Handbook (sample document)

This short handbook is a sample document for testing the Research Analyst's
ingestion and question answering. Every fact below is about this project.

## What RAG is

Retrieval-augmented generation (RAG) answers a question in two steps. First a
retriever finds the passages most relevant to the question in a document
collection. Then a large language model writes an answer using only those
passages, citing them. Because the model is grounded in retrieved text, RAG
reduces hallucination and lets the system answer questions about private
documents the model was never trained on.

## Chunking

Documents are split into chunks before indexing. This project uses a recursive
character splitter with a chunk size of 1000 characters and an overlap of 150
characters. The overlap keeps sentences that straddle a boundary retrievable
from both neighbouring chunks. PDF files are loaded one page at a time so that
every chunk remembers its page number, which is shown in citations.

## Embeddings

Each chunk is turned into a vector by the nomic-embed-text model, running
locally in Ollama. The model produces 768-dimensional vectors. It expects the
prefix "search_document:" on indexed text and "search_query:" on questions;
the project adds these prefixes automatically.

## Vector store

Vectors are stored in Qdrant running in local mode, which keeps everything in
the data/qdrant folder without a separate server. Qdrant was chosen over
Chroma because it is a much smaller download and because it supports hybrid
search, which Stage 2 of the roadmap will use. Local mode allows only one
process to open the folder at a time.

## Retrieval and answering

For each question the retriever returns the top 4 most similar chunks by cosine
similarity. The language model, llama3.2:3b by default, receives these chunks
as numbered passages and must cite them inline like [1] or [2]. If the passages
do not contain the answer, it replies that it could not find the information.

## Roadmap

Stage 2 adds hybrid search, combining BM25 keyword search with vector search,
plus a cross-encoder re-ranker. Stage 3 adds LangGraph agents: a planner, a
researcher, a fact-checker and a writer, with a human approval step. Stage 4
measures quality with RAGAS and traces runs with Langfuse.
