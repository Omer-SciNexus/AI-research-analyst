"""Stage 3 UI: multi-step research with a human approval step."""

import uuid

import streamlit as st
from langgraph.types import Command

from research_analyst.agent import initial_state, run_config
from research_analyst.ui import get_research_graph

st.set_page_config(page_title="Research Agent", page_icon="🔬", layout="wide")

STEPS = {
    "plan": "🗺️ Planned sub-questions",
    "research": "🔎 Researched a sub-question",
    "write": "✍️ Wrote the report",
    "fact_check": "✅ Fact-checked the summary",
}

graph = get_research_graph()
ss = st.session_state


def config() -> dict:
    return run_config(ss.research_thread)


def run(payload) -> None:
    with st.status("Agent working… (a local 3B model on CPU takes a few minutes)", expanded=True) as status:
        for update in graph.stream(payload, config(), stream_mode="updates"):
            for node in update:
                if node in STEPS:
                    status.write(STEPS[node])
        status.update(label="Done", state="complete", expanded=False)
    st.rerun()


st.title("🔬 Research Agent")
st.caption(
    "Plans sub-questions → you approve → researches each in your documents (and optionally the web) "
    "→ writes a cited report → fact-checks it."
)

with st.form("start"):
    question = st.text_area("Research question", placeholder="e.g. How does hybrid search improve RAG answers?")
    use_web = st.toggle("Also search the web (DuckDuckGo)", value=False)
    if st.form_submit_button("Plan research", type="primary") and question.strip():
        ss.research_thread = str(uuid.uuid4())
        run(initial_state(question.strip(), use_web=use_web))

if "research_thread" not in ss:
    st.stop()

snapshot = graph.get_state(config())

if snapshot.interrupts:
    st.subheader("Review the plan")
    st.caption("Edit the sub-questions (one per line) and approve, or ask the planner to try again.")
    plan_text = st.text_area("Sub-questions", "\n".join(snapshot.interrupts[0].value["plan"]), height=150)
    approve_col, revise_col = st.columns(2)
    with approve_col:
        if st.button("✅ Approve and research", type="primary", width="stretch"):
            run(Command(resume={"action": "approve", "plan": plan_text.splitlines()}))
    with revise_col:
        feedback = st.text_input("Feedback for the planner", placeholder="e.g. focus on costs")
        if st.button("🔁 Revise plan", width="stretch", disabled=not feedback.strip()):
            run(Command(resume={"action": "revise", "feedback": feedback.strip()}))

elif report := snapshot.values.get("report"):
    issues = snapshot.values.get("issues") or []
    if issues:
        st.warning(f"Fact-check flagged {len(issues)} issue(s); see the end of the report.")
    else:
        st.success("Fact-check passed.")
    st.download_button("⬇️ Download report (.md)", report, file_name="research-report.md", mime="text/markdown")
    st.markdown(report)
    with st.expander("Source passages"):
        for i, source in enumerate(snapshot.values.get("sources", []), start=1):
            st.markdown(f"**[{i}] {source['label']}**" + (f" — {source['url']}" if source["url"] else ""))
            st.caption(source["text"][:600])
