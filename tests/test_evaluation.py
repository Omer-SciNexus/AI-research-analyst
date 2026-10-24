from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_analyst.evaluation import (
    EvalCase,
    first_relevant_rank,
    is_refusal,
    is_relevant,
    lexical_grounding,
    load_eval_set,
)
from research_analyst.tracing import JsonlTracer, load_records

CASE = EvalCase(question="q", source="h.md", section="Chunking", keywords=["1000"])


def doc(source: str, section: str = "") -> Document:
    return Document(page_content="x", metadata={"source": source, "section": section})


def test_relevance_matches_source_and_last_heading():
    assert is_relevant(doc("h.md", "Handbook > Chunking"), CASE)
    assert not is_relevant(doc("h.md", "Handbook > Embeddings"), CASE)
    assert not is_relevant(doc("other.md", "Glossary > Chunking"), CASE)


def test_first_relevant_rank():
    docs = [doc("h.md", "H > Embeddings"), doc("h.md", "H > Chunking"), doc("h.md", "H > Chunking")]
    assert first_relevant_rank(docs, CASE) == 2
    assert first_relevant_rank(docs[:1], CASE) is None


def test_refusal_detection():
    assert is_refusal("I couldn't find this in the indexed documents.")
    assert not is_refusal("The chunk size is 1000 characters [1].")


def test_lexical_grounding_penalises_invented_words():
    docs = [Document(page_content="The model produces 768-dimensional vectors stored in Qdrant.")]
    assert lexical_grounding("According to [1], the vectors have 768 dimensions.", docs) > 0.6
    assert lexical_grounding("The vectors live in Pinecone with 1536 dimensions [1].", docs) < 0.5
    assert lexical_grounding("[1]", docs) is None


def test_eval_set_is_well_formed():
    documents, cases = load_eval_set()
    assert all(path.exists() for path in documents)
    assert any(not c.answerable for c in cases), "needs unanswerable questions for refusal accuracy"
    assert all(c.keywords for c in cases if c.answerable)


def test_tracer_records_run_and_llm_call(tmp_path):
    tracer = JsonlTracer(tmp_path)
    llm = FakeListChatModel(responses=["pong"])
    (llm | (lambda m: m.content)).invoke("ping", config={"callbacks": [tracer], "run_name": "smoke"})

    records = load_records(tmp_path)
    trace = next(r for r in records if r["kind"] == "trace")
    call = next(r for r in records if r["kind"] == "llm")
    assert trace["name"] == "smoke"
    assert call["trace_id"] == trace["trace_id"]
    assert call["output_chars"] == 4 and call["duration_s"] >= 0
