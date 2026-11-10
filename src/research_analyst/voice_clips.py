"""Find one speaker's clips across videos, for training a Piper voice.

The speaker marks each usable clip as theirs or not by ear. Speaker embeddings only decide
the order the clips are shown in (most like the reference first): they also react to the
microphone, room and call audio, so on a screen-recorded call other people can score as
high as the speaker, and two voices end up in one group.

Files under the output directory:
    cache/<source id>.json/.npy  sentence clips and their speaker embeddings (slow to make)
    cache/sources.json           the videos, and which are references (only the speaker in them)
    candidates/, candidates.json every usable clip, waiting to be marked
    labels.json                  the speaker's answer per clip
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from research_analyst.voice_dataset import Utterance

MIN_CONFIDENCE = 0.75  # mean Whisper word probability; below it the transcript is likely wrong
MIN_WORD_CONFIDENCE = 0.2
MINE, NOT_MINE, UNSURE = "mine", "not mine", "unsure"


@dataclass
class Source:
    path: Path
    is_reference: bool
    utterances: list[Utterance]
    embeddings: np.ndarray

    @property
    def id(self) -> str:
        return source_id(self.path)

    def clip_name(self, index: int) -> str:
        return f"{self.id}_{index + 1:04d}.wav"


@dataclass
class Candidate:
    clip: str
    source: str
    reference: bool
    start: float
    seconds: float
    score: float  # similarity to the reference videos' voice, only for ordering
    text: str


def source_id(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")[:16]
    return f"{slug}_{hashlib.sha1(path.name.encode()).hexdigest()[:6]}"


def confident(utterance: Utterance) -> bool:
    return utterance.confidence >= MIN_CONFIDENCE and utterance.min_confidence >= MIN_WORD_CONFIDENCE


def centroid(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    return mean / np.linalg.norm(mean)


def _write_json(path: Path, data) -> None:
    """Replace a file whole, so a crash never leaves half of it behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_analysis(cache_dir: Path, path: Path, utterances: list[Utterance], embeddings: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / source_id(path)
    np.save(stem.with_suffix(".npy"), embeddings)
    # Written last: its presence marks the cache as complete.
    stem.with_suffix(".json").write_text(json.dumps([asdict(u) for u in utterances]), encoding="utf-8")


def load_analysis(cache_dir: Path, path: Path) -> tuple[list[Utterance], np.ndarray] | None:
    stem = cache_dir / source_id(path)
    if not stem.with_suffix(".json").exists():
        return None
    utterances = [Utterance(**u) for u in json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))]
    return utterances, np.load(stem.with_suffix(".npy"))


def save_manifest(output_dir: Path, references: list[Path], sources: list[Path]) -> None:
    entries = [{"path": str(p), "reference": True} for p in references]
    entries += [{"path": str(p), "reference": False} for p in sources if p not in references]
    _write_json(output_dir / "cache" / "sources.json", entries)


def load_sources(output_dir: Path) -> list[Source]:
    """Every analysed video from the manifest."""
    manifest = output_dir / "cache" / "sources.json"
    if not manifest.exists():
        return []
    sources = []
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        path = Path(entry["path"])
        analysis = load_analysis(output_dir / "cache", path)
        if analysis is not None:
            sources.append(Source(path, entry["reference"], *analysis))
    return sources


def reference_profile(sources: list[Source]) -> np.ndarray:
    return centroid(np.concatenate([s.embeddings for s in sources if s.is_reference and len(s.embeddings)]))


def save_candidates(output_dir: Path, candidates: list[Candidate]) -> None:
    _write_json(output_dir / "candidates.json", [asdict(c) for c in candidates])


def load_candidates(output_dir: Path) -> list[Candidate]:
    path = output_dir / "candidates.json"
    return [Candidate(**c) for c in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []


def load_labels(output_dir: Path, candidates: list[Candidate]) -> dict[str, str]:
    """The speaker's answer per clip. An answer counts only while the clip still has the text
    it was given for: a re-cut clip under the same name may hold someone else's words."""
    path = output_dir / "labels.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    text = {c.clip: c.text for c in candidates}
    return {clip: entry["label"] for clip, entry in saved.items() if text.get(clip) == entry["text"]}


def save_label(output_dir: Path, candidate: Candidate, label: str) -> None:
    path = output_dir / "labels.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    saved[candidate.clip] = {"label": label, "text": candidate.text}
    _write_json(path, saved)


def is_mine(candidate: Candidate, labels: dict[str, str]) -> bool:
    """Reference videos have only the speaker in them; elsewhere, only clips they marked."""
    return candidate.reference or labels.get(candidate.clip) == MINE
