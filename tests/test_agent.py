"""Research agent tests with the LLM and search replaced by fakes (no Ollama needed)."""

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_analyst.agent import (
    Finding,
    ResearchAgent,
    Source,
    Verdict,
    citation_issues,
    initial_state,
    merge_findings,
)


def src(label: str, text: str = "") -> Source:
    return Source(label=label, text=text or f"text of {label}", url=None)


def test_merge_findings_dedupes_sources_and_renumbers_citations():
    findings = [
        Finding(sub_question="q1", answer="A [1] and B [2].", sources=[src("a"), src("b")]),
        Finding(sub_question="q2", answer="B again [1], C [2], bogus [9].", sources=[src("b"), src("c")]),
    ]
    sources, renumbered = merge_findings(findings)
    assert [s["label"] for s in sources] == ["a", "b", "c"]
    assert renumbered[0]["answer"] == "A [1] and B [2]."
    assert renumbered[1]["answer"] == "B again [2], C [3], bogus ."


def test_citation_issues():
    assert citation_issues("Fine [1][2].", 2) == []
    assert citation_issues("Out of range [3].", 2) == ["Citation [3] does not match any source."]
    assert citation_issues("No citations.", 2) == ["The summary cites no sources."]


class FakeAgent(ResearchAgent):
    """Deterministic stand-ins for every LLM / search call."""

    def __init__(self, verdicts: list[bool]):
        super().__init__(store=None)
        self.verdicts = verdicts
        self.summaries = 0
        self.plans: list[str] = []

    def make_plan(self, question, feedback):
        self.plans.append(feedback)
        return ["What is X?", "Why does X matter?"] if not feedback else ["What is X exactly?"]

    def gather(self, sub_question, use_web):
        return [Document(page_content=f"Facts about {sub_question}", metadata={"source": "doc.md"})]

    def answer(self, sub_question, docs):
        return f"Answer to {sub_question} [1]."

    def summarize(self, question, findings, issues):
        self.summaries += 1
        return "X is explained here [1]." if not issues else "X is explained, corrected [1]."

    def verify(self, summary, sources):
        ok = self.verdicts.pop(0)
        return Verdict(supported=ok, unsupported_claims=[] if ok else ["X is explained here"])


def run_until_pause(graph, payload, config):
    for _ in graph.stream(payload, config, stream_mode="updates"):
        pass
    return graph.get_state(config)


def test_graph_pauses_for_plan_approval_then_writes_report():
    agent = FakeAgent(verdicts=[True])
    graph = agent.build(InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    state = run_until_pause(graph, initial_state("What is X?"), config)
    assert state.interrupts and state.interrupts[0].value["plan"] == ["What is X?", "Why does X matter?"]
    assert "report" not in state.values  # nothing researched before approval

    state = run_until_pause(graph, Command(resume={"action": "approve"}), config)
    report = state.values["report"]
    assert not state.interrupts
    assert "## Summary" in report and "### Why does X matter?" in report
    assert "All summary claims are supported" in report
    assert len(state.values["sources"]) == 2  # one per sub-question


def test_reviewer_can_revise_and_edit_the_plan():
    agent = FakeAgent(verdicts=[True])
    graph = agent.build(InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    run_until_pause(graph, initial_state("What is X?"), config)
    state = run_until_pause(graph, Command(resume={"action": "revise", "feedback": "only one question"}), config)
    assert "only one question" in agent.plans[-1]
    assert state.values["plan"] == ["What is X exactly?"]

    edited = ["What is X exactly?", "  ", "Who uses X?"]
    state = run_until_pause(graph, Command(resume={"action": "approve", "plan": edited}), config)
    assert state.values["plan"] == ["What is X exactly?", "Who uses X?"]
    assert sorted(f["sub_question"] for f in state.values["findings"]) == sorted(state.values["plan"])
    assert state.values["report"].index("### What is X exactly?") < state.values["report"].index("### Who uses X?")


def test_unsupported_summary_is_rewritten_once():
    agent = FakeAgent(verdicts=[False, False])
    graph = agent.build(InMemorySaver())
    config = {"configurable": {"thread_id": "t3"}}

    run_until_pause(graph, initial_state("What is X?"), config)
    state = run_until_pause(graph, Command(resume={"action": "approve"}), config)
    assert agent.summaries == 2  # original + one revision, then stop
    assert state.values["revisions"] == 1
    assert "Could not verify" in state.values["report"]
