# RAG Glossary (distractor document)

A general glossary of retrieval terms. It shares vocabulary with the RAG
Handbook but describes the field in general, not this project, so the
evaluation can check that retrieval finds the project-specific passage.

## Large language models

A large language model predicts the next token of text. Popular open models
include Llama, Mistral, Qwen and Gemma, released in sizes from about one
billion to hundreds of billions of parameters. Smaller models run on laptops,
while larger ones need GPUs with plenty of memory.

## Hallucination

A hallucination is a fluent but false statement produced by a language model.
Common mitigations are lowering the temperature, asking the model to say when
it does not know, and grounding answers in retrieved documents.

## Text chunking strategies

Long documents are usually divided before indexing. Fixed-size chunking cuts
every few hundred tokens; sentence chunking keeps whole sentences together;
semantic chunking starts a new piece when the topic shifts. Typical sizes range
from 256 to 2048 tokens, and teams tune them per corpus.

## Vector embeddings

An embedding model maps text to a list of numbers so that similar meanings land
close together. Common sizes are 384, 1024 and 1536 dimensions. Well-known
families include Sentence-Transformers, E5, BGE and OpenAI's text-embedding
models.

## Vector databases

A vector database stores embeddings and finds nearest neighbours quickly using
indexes such as HNSW. Popular options are Pinecone, Weaviate, Milvus, Chroma,
pgvector and Qdrant. Managed cloud versions are billed by storage and usage.

## Keyword search and BM25

BM25 ranks documents by how often query terms appear, discounted for very
common words and adjusted for document length. It excels at exact names, codes
and numbers that embedding models sometimes blur.

## Re-rankers

A re-ranker is a cross-encoder that reads the query and a candidate passage
together and outputs a relevance score. It is slower than vector search, so it
is applied only to a short list of candidates.

## Agents and orchestration

An agent lets a language model decide which tools to call and in what order.
Frameworks such as LangGraph, CrewAI and AutoGen add state, loops and human
approval steps on top of plain model calls.

## Evaluation and observability

RAG systems are measured on retrieval quality (hit rate, MRR) and answer
quality (faithfulness, relevance). Tracing tools record every model call with
its latency and token usage so regressions can be found.
