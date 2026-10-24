"""Stage 6: REST API + web UI.

Run:  uv run uvicorn research_analyst.api:app --port 8000   →  http://localhost:8000
Docs: http://localhost:8000/docs

Endpoints (all JSON unless noted):
    GET    /api/health                      status, models, number of indexed chunks
    GET    /api/documents                   {file: chunks}
    POST   /api/documents                   multipart upload (pdf/md/txt) → indexed
    DELETE /api/documents/{name}            remove a file from the index
    POST   /api/ask                         cited answer + confidence + sources
    POST   /api/ask/stream                  same, as server-sent events (meta, token…, done)
    POST   /api/research                    start the research agent → plan awaiting approval
    POST   /api/research/{id}/resume        approve / edit / revise the plan → report
    GET    /api/research/{id}               current state of a run

Like the Streamlit app, it opens data/qdrant in local mode: run one of them at a time.
"""

import json
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_analyst.agent import WEB_MODES, ResearchAgent, default_web_mode, get_checkpointer, initial_state, run_config
from research_analyst.confidence import load_model
from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.ingest import SUPPORTED_SUFFIXES, find_files, ingest_file
from research_analyst.rag import NOT_FOUND, generate_answer, prepare, source_label, stream_answer
from research_analyst.vectorstore import MODES, delete_source, get_vectorstore, list_sources

WEB_DIR = PROJECT_ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "store"):  # tests inject their own store
        app.state.store = get_vectorstore()
        if settings.index_samples_on_start and not list_sources(app.state.store):
            for path in find_files(PROJECT_ROOT / "samples"):  # demo content for a fresh deployment
                ingest_file(path, app.state.store)
    if not hasattr(app.state, "graph"):
        app.state.graph = ResearchAgent(app.state.store).build(get_checkpointer())
    yield


app = FastAPI(title="Research Analyst API", version="1.0.0", lifespan=lifespan)


def get_store(request: Request) -> QdrantVectorStore:
    return request.app.state.store


def get_graph(request: Request):
    return request.app.state.graph


# ── Schemas ──────────────────────────────────────────────────────────────────


class SourceOut(BaseModel):
    label: str
    text: str
    url: str | None = None
    rerank_score: float | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool | None = None
    skip_low_confidence: bool | None = None


class AskOut(BaseModel):
    answer: str
    confidence: float | None
    threshold: float | None
    llm_skipped: bool
    sources: list[SourceOut]


class ResearchIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    web: Literal["off", "auto", "always"] | None = None


class ResumeIn(BaseModel):
    action: Literal["approve", "revise"]
    plan: list[str] | None = None
    feedback: str | None = None


class ResearchOut(BaseModel):
    thread_id: str
    status: Literal["awaiting_approval", "done", "not_found"]
    plan: list[str] = []
    report: str | None = None
    issues: list[str] = []
    sources: list[SourceOut] = []


def to_source(doc: Document) -> SourceOut:
    return SourceOut(
        label=source_label(doc), text=doc.page_content, url=doc.metadata.get("url"), rerank_score=doc.metadata.get("rerank_score")
    )


# ── Health & documents ───────────────────────────────────────────────────────


@app.get("/api/health")
def health(store: QdrantVectorStore = Depends(get_store)) -> dict:
    model = load_model()
    return {
        "status": "ok",
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "embeddings": f"{settings.embedding_provider}/{settings.embedding_model}",
        "retrieval": {"mode": settings.retrieval_mode, "rerank": settings.rerank, "modes": list(MODES)},
        "confidence_model": {"name": model.name, "threshold": model.threshold} if model else None,
        "web_modes": list(WEB_MODES),
        "default_web_mode": default_web_mode(),
        "chunks": sum(list_sources(store).values()),
    }


@app.get("/api/documents")
def documents(store: QdrantVectorStore = Depends(get_store)) -> dict[str, int]:
    return list_sources(store)


@app.post("/api/documents")
def upload(files: list[UploadFile], store: QdrantVectorStore = Depends(get_store)) -> dict[str, int]:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    indexed = {}
    for file in files:
        name = Path(file.filename or "").name  # never trust client paths
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(415, f"{name}: only {', '.join(sorted(SUPPORTED_SUFFIXES))} files are supported")
        path = settings.docs_dir / name
        path.write_bytes(file.file.read())
        indexed[name] = ingest_file(path, store)
    return indexed


@app.delete("/api/documents/{name}")
def remove(name: str, store: QdrantVectorStore = Depends(get_store)) -> dict:
    if name not in list_sources(store):
        raise HTTPException(404, f"{name} is not indexed")
    delete_source(store, name)
    return {"removed": name}


# ── Chat ─────────────────────────────────────────────────────────────────────


@app.post("/api/ask")
def ask(body: AskIn, store: QdrantVectorStore = Depends(get_store)) -> AskOut:
    turn = prepare(body.question, store, mode=body.mode, use_rerank=body.rerank, skip_low_confidence=body.skip_low_confidence)
    answer = NOT_FOUND if turn.skip_llm else generate_answer(body.question, turn.docs)
    return AskOut(
        answer=answer,
        confidence=turn.confidence,
        threshold=turn.threshold,
        llm_skipped=turn.skip_llm,
        sources=[to_source(d) for d in turn.docs],
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(body: AskIn, store: QdrantVectorStore = Depends(get_store)) -> StreamingResponse:
    turn = prepare(body.question, store, mode=body.mode, use_rerank=body.rerank, skip_low_confidence=body.skip_low_confidence)

    def events() -> Iterator[str]:
        meta = {
            "confidence": turn.confidence,
            "threshold": turn.threshold,
            "llm_skipped": turn.skip_llm,
            "sources": [to_source(d).model_dump() for d in turn.docs],
        }
        yield _sse("meta", meta)
        try:
            for token in [NOT_FOUND] if turn.skip_llm else stream_answer(body.question, turn.docs):
                yield _sse("token", token)
        except Exception as exc:  # the client sees why the answer stopped
            yield _sse("error", str(exc))
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")


# ── Research agent ───────────────────────────────────────────────────────────


def _research_state(graph, thread_id: str) -> ResearchOut:
    snapshot = graph.get_state(run_config(thread_id))
    if not snapshot.values:
        return ResearchOut(thread_id=thread_id, status="not_found")
    if snapshot.interrupts:
        return ResearchOut(thread_id=thread_id, status="awaiting_approval", plan=snapshot.interrupts[0].value["plan"])
    values = snapshot.values
    return ResearchOut(
        thread_id=thread_id,
        status="done",
        plan=values.get("plan", []),
        report=values.get("report"),
        issues=values.get("issues", []),
        sources=[SourceOut(label=s["label"], text=s["text"], url=s["url"]) for s in values.get("sources", [])],
    )


def _run(graph, payload, thread_id: str) -> None:
    for _ in graph.stream(payload, run_config(thread_id), stream_mode="updates"):
        pass


@app.post("/api/research")
def start_research(body: ResearchIn, graph=Depends(get_graph)) -> ResearchOut:
    thread_id = str(uuid.uuid4())
    _run(graph, initial_state(body.question, web=body.web), thread_id)
    return _research_state(graph, thread_id)


@app.post("/api/research/{thread_id}/resume")
def resume_research(thread_id: str, body: ResumeIn, graph=Depends(get_graph)) -> ResearchOut:
    state = _research_state(graph, thread_id)
    if state.status != "awaiting_approval":
        raise HTTPException(409, f"run {thread_id} is {state.status}, not awaiting approval")
    _run(graph, Command(resume=body.model_dump(exclude_none=True)), thread_id)
    return _research_state(graph, thread_id)


@app.get("/api/research/{thread_id}")
def research_status(thread_id: str, graph=Depends(get_graph)) -> ResearchOut:
    state = _research_state(graph, thread_id)
    if state.status == "not_found":
        raise HTTPException(404, f"no run {thread_id}")
    return state


# ── Web UI ───────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
