"""Run the LangGraph research agent from the command line.

Usage:
    uv run python scripts/research.py "How does hybrid search improve RAG?"
    uv run python scripts/research.py --web "What is LangGraph used for?"
    uv run python scripts/research.py --yes "..."     # approve the plan automatically

The agent pauses after planning so you can approve the plan or ask for changes.
Reports are saved to data/reports/.
"""

import argparse
import re
import sys
import time
import uuid

from langgraph.types import Command

from research_analyst.agent import ResearchAgent, get_checkpointer, initial_state
from research_analyst.config import settings
from research_analyst.vectorstore import get_vectorstore

STEPS = {
    "plan": "planned sub-questions",
    "research": "researched a sub-question",
    "write": "wrote the report",
    "fact_check": "fact-checked the summary",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--web", action="store_true", help="also search the web (DuckDuckGo)")
    parser.add_argument("--yes", action="store_true", help="approve the plan without asking")
    args = parser.parse_args()

    question = " ".join(args.question)
    graph = ResearchAgent(get_vectorstore()).build(get_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload = initial_state(question, use_web=args.web)
    start = time.perf_counter()

    while True:
        for update in graph.stream(payload, config, stream_mode="updates"):
            for node in update:
                if node in STEPS:
                    print(f"  [{time.perf_counter() - start:5.0f}s] {STEPS[node]}")
        state = graph.get_state(config)
        if not state.interrupts:
            break

        plan = state.interrupts[0].value["plan"]
        print("\nProposed plan:")
        for i, sub_question in enumerate(plan, start=1):
            print(f"  {i}. {sub_question}")
        reply = "" if args.yes else input("\nPress Enter to approve, or type feedback to revise: ").strip()
        payload = Command(resume={"action": "revise", "feedback": reply} if reply else {"action": "approve"})

    report = state.values["report"]
    print("\n" + report)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "report"
    path = settings.reports_dir / f"{slug}.md"
    path.write_text(report, encoding="utf-8")
    print(f"[saved to {path} in {time.perf_counter() - start:.0f}s]")


if __name__ == "__main__":
    main()
