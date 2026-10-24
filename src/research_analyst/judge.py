"""LLM-as-judge: is a text fully supported by its source passages?

Every check is a one-word YES/NO question (for the whole text, and per sentence to
find the unsupported claims): small models handle that far better than structured
JSON verdicts, which llama3.2:3b answered `false` even for exact quotes.

Even so, a 3B judge is unreliable on real answers (it rejected most correct answers
in the evaluation set), so the judge model is configured separately via
JUDGE_PROVIDER / JUDGE_MODEL; a larger model, e.g. on Groq's free tier, is advised.
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from research_analyst.config import settings
from research_analyst.llm import get_llm, with_retry

SUPPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check a text against source passages. Reply with exactly one word: YES if every "
            "statement in the TEXT is stated in the PASSAGES (numbers must match exactly), otherwise NO.",
        ),
        ("human", "PASSAGES:\n{passages}\n\nTEXT:\n{text}\n\nIs every statement supported? YES or NO:"),
    ]
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def is_supported(text: str, passages: str) -> bool | None:
    """True / False, or None if the judge's reply was neither YES nor NO."""
    judge = get_llm(model=settings.judge_model, provider=settings.judge_provider, temperature=0)
    reply = with_retry(SUPPORT_PROMPT | judge).invoke({"passages": passages, "text": text})
    word = reply.content.strip().strip(".*\"'").upper()
    if word.startswith("YES"):
        return True
    if word.startswith("NO"):
        return False
    return None


def unsupported_claims(text: str, passages: str) -> list[str]:
    """The sentences of `text` the judge says the passages don't support."""
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    return [s for s in sentences if is_supported(s, passages) is False]
