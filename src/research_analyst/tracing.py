"""Local tracing: every LLM call and top-level run is appended to data/traces/<date>.jsonl.

Pass `get_callbacks()` in a runnable's config:  chain.invoke(x, config={"callbacks": get_callbacks()}).
Nested calls (LangGraph nodes, parallel tasks) inherit it automatically.

Optional Langfuse (free cloud tier, no Docker): `uv add langfuse`, then set
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST; runs are sent there too.
"""

import json
import os
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from research_analyst.config import settings


class JsonlTracer(BaseCallbackHandler):
    """Writes one JSON line per finished LLM call ("llm") and per finished top-level run ("trace")."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._open: dict[UUID, dict] = {}
        self._root: dict[UUID, UUID] = {}  # run -> top-level run it belongs to

    # bookkeeping

    def _root_of(self, run_id: UUID, parent_run_id: UUID | None) -> UUID:
        root = self._root.get(parent_run_id, parent_run_id) if parent_run_id else run_id
        self._root[run_id] = root
        return root

    def _finish(self, run_id: UUID, **fields: Any) -> None:
        with self._lock:
            record = self._open.pop(run_id, None)
            if record is None:
                return
            record.update(fields, duration_s=round(time.time() - record["start"], 3))
            if record["kind"] == "trace":  # forget the finished tree
                self._root = {r: root for r, root in self._root.items() if root != run_id}
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{datetime.now():%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    # chains: only top-level runs are recorded

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kwargs):
        with self._lock:
            root = self._root_of(run_id, parent_run_id)
            if parent_run_id is None:
                self._open[run_id] = {
                    "kind": "trace",
                    "trace_id": str(root),
                    "name": name or (serialized or {}).get("name", "run"),
                    "start": time.time(),
                }

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, error=repr(error))

    # LLM calls

    def _llm_start(self, run_id, parent_run_id, input_chars, kwargs):
        metadata = kwargs.get("metadata") or {}
        with self._lock:
            self._open[run_id] = {
                "kind": "llm",
                "trace_id": str(self._root_of(run_id, parent_run_id)),
                "name": metadata.get("langgraph_node") or kwargs.get("name") or "llm",
                "model": metadata.get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model"),
                "input_chars": input_chars,
                "start": time.time(),
            }

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        chars = sum(len(str(m.content)) for batch in messages for m in batch)
        self._llm_start(run_id, parent_run_id, chars, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._llm_start(run_id, parent_run_id, sum(map(len, prompts)), kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
        self._finish(
            run_id,
            output_chars=len(generation.text) if generation else 0,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, error=repr(error))


@lru_cache
def get_callbacks() -> list[BaseCallbackHandler]:
    callbacks: list[BaseCallbackHandler] = [JsonlTracer(settings.traces_dir)]
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            pass  # keys set but package not installed: local tracing only
        else:
            callbacks.append(CallbackHandler())
    return callbacks


def load_records(directory: Path | None = None, days: int = 7) -> list[dict]:
    """Records from the newest `days` trace files, newest first."""
    directory = Path(directory or settings.traces_dir)
    records = []
    for path in sorted(directory.glob("*.jsonl"), reverse=True)[:days]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda r: r["start"], reverse=True)
