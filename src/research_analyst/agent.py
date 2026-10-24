"""Stage 3: LangGraph research agent.

    plan ──► review ──► research ×N ──► write ──► fact_check ──► END
     ▲         │ (human)   (parallel)     ▲            │
     └─────────┘ revise                   └────────────┘ unsupported claims (1 retry)

- plan:       the LLM splits the question into 2-4 sub-questions
- review:     the graph pauses (interrupt) so a human can approve, edit or reject the plan
- research:   one task per sub-question (Send fan-out): hybrid search over the documents,
              plus DuckDuckGo when web mode is "always", or "auto" and the confidence
              model (Stage 5) says the documents can't answer; then a cited answer
- write:      merges all sources into one numbered list, renumbers citations and adds an
              executive summary
- fact_check: checks citations and asks the LLM whether the summary is supported by the
              cited sources; sends it back to write once if not

Every LLM call lives in its own small method, so tests can replace them.
"""

import operator
import re
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field

from research_analyst.confidence import assess
from research_analyst.config import settings
from research_analyst.judge import is_supported, unsupported_claims
from research_analyst.llm import get_llm, with_retry
from research_analyst.rag import generate_answer, source_label
from research_analyst.tracing import get_callbacks
from research_analyst.web import web_search

WEB_MODES = ("off", "auto", "always")
MAX_SUB_QUESTIONS = 4
MAX_REVISIONS = 1
CITATION = re.compile(r"\[(\d+)\]")


# ── State ────────────────────────────────────────────────────────────────────


class Source(TypedDict):
    label: str
    text: str
    url: str | None


class Finding(TypedDict):
    sub_question: str
    answer: str
    sources: list[Source]


class ResearchState(TypedDict, total=False):
    question: str
    web: str  # one of WEB_MODES
    plan: list[str]
    plan_feedback: str
    findings: Annotated[list[Finding], operator.add]
    sources: list[Source]  # merged, numbered list the report cites
    summary: str
    report: str
    issues: list[str]
    revisions: int


class ResearchTask(TypedDict):
    sub_question: str
    web: str


# ── Structured LLM outputs ───────────────────────────────────────────────────


class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="2 to 4 short, focused sub-questions that together answer the main question"
    )


class Verdict(BaseModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan research. Break the user's question into 2 to 4 short, focused sub-questions "
            "that can each be answered from documents or a web search. Do not answer them.",
        ),
        ("human", "Question: {question}{feedback}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write the executive summary of a research report in 3 to 5 sentences. Use ONLY the "
            "findings provided. Keep their citation numbers exactly as written, like [2]. If the "
            "findings could not answer part of the question, say so.",
        ),
        ("human", "Main question: {question}\n\nFindings:\n\n{findings}{issues}"),
    ]
)


# ── Helpers (pure, unit-tested) ──────────────────────────────────────────────


def to_source(doc: Document) -> Source:
    return Source(label=source_label(doc), text=doc.page_content, url=doc.metadata.get("url"))


def merge_findings(findings: list[Finding]) -> tuple[list[Source], list[Finding]]:
    """Merge every finding's sources into one deduplicated list and renumber citations to match."""
    merged: list[Source] = []
    index: dict[tuple[str, str], int] = {}
    renumbered: list[Finding] = []
    for finding in findings:
        local_to_global: dict[int, int] = {}
        for local, source in enumerate(finding["sources"], start=1):
            key = (source["label"], source["text"][:200])
            if key not in index:
                merged.append(source)
                index[key] = len(merged)
            local_to_global[local] = index[key]
        answer = CITATION.sub(
            lambda m: f"[{local_to_global[int(m.group(1))]}]" if int(m.group(1)) in local_to_global else "",
            finding["answer"],
        )
        renumbered.append(Finding(sub_question=finding["sub_question"], answer=answer, sources=finding["sources"]))
    return merged, renumbered


def citation_issues(text: str, source_count: int) -> list[str]:
    cited = {int(n) for n in CITATION.findall(text)}
    issues = [f"Citation [{n}] does not match any source." for n in sorted(cited) if not 1 <= n <= source_count]
    if source_count and not cited:
        issues.append("The summary cites no sources.")
    return issues


def render_report(question: str, summary: str, findings: list[Finding], sources: list[Source]) -> str:
    sections = "\n\n".join(f"### {f['sub_question']}\n\n{f['answer']}" for f in findings)
    source_lines = "\n".join(
        f"{i}. {s['label']}" + (f" — {s['url']}" if s["url"] else "") for i, s in enumerate(sources, start=1)
    )
    return f"# {question}\n\n## Summary\n\n{summary}\n\n## Findings\n\n{sections}\n\n## Sources\n\n{source_lines}\n"


# ── Agent ────────────────────────────────────────────────────────────────────


class ResearchAgent:
    def __init__(self, store: QdrantVectorStore):
        self.store = store

    # LLM / IO calls (overridden in tests)

    def make_plan(self, question: str, feedback: str) -> list[str]:
        try:
            plan = with_retry(PLAN_PROMPT | get_llm(temperature=0).with_structured_output(Plan)).invoke(
                {"question": question, "feedback": feedback}
            )
            return plan.sub_questions
        except Exception:  # small models sometimes emit invalid JSON: fall back to the question itself
            return []

    def gather(self, sub_question: str, web: str) -> list[Document]:
        assessment = assess(sub_question, self.store)
        docs = assessment.docs[:3]
        if web == "always" or (web == "auto" and not assessment.answerable):
            docs += web_search(sub_question, k=3)
        return docs

    def answer(self, sub_question: str, docs: list[Document]) -> str:
        return generate_answer(sub_question, docs)

    def summarize(self, question: str, findings: str, issues: list[str]) -> str:
        fix = ("\n\nYour previous summary had unsupported claims; remove or correct them:\n- " + "\n- ".join(issues)) if issues else ""
        chain = with_retry(SUMMARY_PROMPT | get_llm())
        return chain.invoke({"question": question, "findings": findings, "issues": fix}).content

    def verify(self, summary: str, sources: str) -> Verdict:
        if is_supported(summary, sources) is not False:  # YES, or an unclear reply: don't block the report
            return Verdict(supported=True)
        return Verdict(supported=False, unsupported_claims=unsupported_claims(summary, sources))

    # Nodes

    def plan(self, state: ResearchState) -> dict:
        feedback = ""
        if state.get("plan_feedback"):
            previous = "\n".join(f"- {q}" for q in state.get("plan", []))
            feedback = f"\n\nPrevious plan:\n{previous}\n\nReviewer feedback: {state['plan_feedback']}"
        plan = [q.strip() for q in self.make_plan(state["question"], feedback) if q.strip()]
        return {"plan": plan[:MAX_SUB_QUESTIONS] or [state["question"]]}

    def review(self, state: ResearchState) -> Command[Literal["plan", "research"]]:
        decision = interrupt({"question": state["question"], "plan": state["plan"]})
        if decision.get("action") == "revise":
            return Command(goto="plan", update={"plan_feedback": decision.get("feedback", "")})
        plan = [q.strip() for q in decision.get("plan") or state["plan"] if q.strip()] or state["plan"]
        tasks = [Send("research", ResearchTask(sub_question=q, web=state.get("web", "off"))) for q in plan]
        return Command(goto=tasks, update={"plan": plan})

    def research(self, task: ResearchTask) -> dict:
        docs = self.gather(task["sub_question"], task["web"])
        finding = Finding(
            sub_question=task["sub_question"],
            answer=self.answer(task["sub_question"], docs),
            sources=[to_source(d) for d in docs],
        )
        return {"findings": [finding]}

    def write(self, state: ResearchState) -> dict:
        # Keep the plan's order (parallel tasks finish in any order)
        order = {q: i for i, q in enumerate(state["plan"])}
        findings = sorted(state["findings"], key=lambda f: order.get(f["sub_question"], len(order)))
        sources, findings = merge_findings(findings)
        findings_text = "\n\n".join(f"Q: {f['sub_question']}\nA: {f['answer']}" for f in findings)
        summary = self.summarize(state["question"], findings_text, state.get("issues", []))
        return {
            "sources": sources,
            "summary": summary,
            "report": render_report(state["question"], summary, findings, sources),
        }

    def fact_check(self, state: ResearchState) -> Command[Literal["write", "__end__"]]:
        sources, summary = state["sources"], state["summary"]
        issues = citation_issues(summary, len(sources))
        cited = sorted({int(n) for n in CITATION.findall(summary) if 1 <= int(n) <= len(sources)})
        if cited:
            context = "\n\n".join(f"[{n}] {sources[n - 1]['text'][:700]}" for n in cited)
            verdict = self.verify(summary, context)
            if not verdict.supported:
                issues += verdict.unsupported_claims or ["The summary is not fully supported by its sources."]

        revisions = state.get("revisions", 0)
        if issues and revisions < MAX_REVISIONS:
            return Command(goto="write", update={"issues": issues, "revisions": revisions + 1})

        note = (
            "All summary claims are supported by the cited sources."
            if not issues
            else "Could not verify:\n\n" + "\n".join(f"- {i}" for i in issues)
        )
        return Command(goto=END, update={"issues": issues, "report": f"{state['report']}\n## Fact-check\n\n{note}\n"})

    def build(self, checkpointer: BaseCheckpointSaver | None = None):
        graph = StateGraph(ResearchState)
        graph.add_node("plan", self.plan)
        graph.add_node("review", self.review)
        graph.add_node("research", self.research)
        graph.add_node("write", self.write)
        graph.add_node("fact_check", self.fact_check)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "review")
        graph.add_edge("research", "write")
        graph.add_edge("write", "fact_check")
        return graph.compile(checkpointer=checkpointer)


def get_checkpointer() -> SqliteSaver:
    """Runs are saved to SQLite, so a paused run survives an app restart."""
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(settings.checkpoint_db, check_same_thread=False))


def default_web_mode() -> str:
    return "auto" if settings.auto_web_fallback else "off"


def initial_state(question: str, web: str | None = None) -> ResearchState:
    web = web or default_web_mode()
    if web not in WEB_MODES:
        raise ValueError(f"web must be one of {WEB_MODES}, got {web!r}")
    return ResearchState(question=question, web=web, findings=[], revisions=0, issues=[])


def run_config(thread_id: str) -> dict:
    """Graph config: which saved run to continue, plus tracing for every node's LLM calls."""
    return {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks(), "run_name": "research_agent"}
