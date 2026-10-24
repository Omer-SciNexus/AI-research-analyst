"""Free web search via DuckDuckGo (no API key)."""

from ddgs import DDGS
from langchain_core.documents import Document


def web_search(query: str, k: int = 3) -> list[Document]:
    """Top-k result snippets as Documents; empty list if the search fails."""
    try:
        results = DDGS().text(query, max_results=k)
    except Exception:  # network errors / rate limits: research continues with documents only
        return []
    return [
        Document(
            page_content=r["body"],
            metadata={"source": r.get("title") or r.get("href", "web"), "url": r.get("href")},
        )
        for r in results
        if r.get("body")
    ]
