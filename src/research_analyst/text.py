"""Small text helpers shared by evaluation and the confidence model."""

import re

STOPWORDS = set(
    "the a an and or but of to in on at for by with from as is are was were be been it its this that these "
    "those which who what when where why how does do did can could will would should may might has have had "
    "not no yes than then so such into over about each every only also very more most other their there they "
    "them his her our your we you i according states stated passage passages context answer".split()
)
WORD = re.compile(r"[a-z0-9][a-z0-9_./-]*")
CITATION = re.compile(r"\[\d+\]")


def content_words(text: str) -> list[str]:
    """Lower-cased words minus stopwords, citation markers and very short tokens."""
    words = (w.strip("./-") for w in WORD.findall(CITATION.sub(" ", text.lower())))
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def coverage(words: list[str], text: str) -> float | None:
    """Share of `words` that occur in `text` (None when there are no words)."""
    if not words:
        return None
    lowered = text.lower()
    return sum(w in lowered for w in words) / len(words)
