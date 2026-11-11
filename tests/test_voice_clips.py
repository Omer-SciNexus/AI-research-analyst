import csv
import dataclasses
import importlib.util
import json
import sys
import zipfile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.voice_clips import (
    MINE,
    NOT_MINE,
    UNSURE,
    load_candidates,
    load_labels,
    load_sources,
    save_analysis,
    save_label,
    save_manifest,
)
from research_analyst.voice_dataset import SAMPLE_RATE, Utterance, to_wav

SCRIPTS = PROJECT_ROOT / "scripts"
TEXT = "This is a sentence of words."  # 28 characters, a natural pace over about 2.4 s


def voice(direction: int, count: int, seed: int = 0) -> np.ndarray:
    """Unit speaker embeddings scattered tightly around one axis: one voice."""
    rng = np.random.default_rng(seed + direction)
    vectors = np.zeros((count, 16), dtype=np.float32)
    vectors[:, direction] = 1.0
    vectors += rng.normal(0, 0.05, vectors.shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def clip(start: float, confidence: float = 0.95) -> Utterance:
    return Utterance(start, start + 2.4, TEXT, confidence, 0.8)


def tones_wav(path, count: int) -> None:
    """`count` tone bursts of 2 s, at 1-3 s, 4-6 s, ... with near-silence between them."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.0005, SAMPLE_RATE * (3 * count + 1)).astype(np.float32)
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    for i in range(count):
        start = SAMPLE_RATE * (1 + 3 * i)
        audio[start : start + len(t)] += 0.5 * np.sin(2 * np.pi * (200 + 50 * i) * t)
    to_wav(audio, path)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild(output, monkeypatch):
    prepare = load_script("prepare_voice_clips")
    monkeypatch.setattr(prepare, "settings", dataclasses.replace(settings, voice_clips_dir=output))
    monkeypatch.setattr(sys, "argv", ["prepare_voice_clips.py"])
    prepare.main()


@pytest.fixture
def videos(tmp_path, monkeypatch):
    """An analysed reference video with only voice 0, and a call where voices 1 and 0 speak
    twice each, plus a fifth clip whose transcript is unsure."""
    reference, call = tmp_path / "intro.wav", tmp_path / "call.wav"
    tones_wav(reference, 1)
    tones_wav(call, 5)
    output = tmp_path / "voice_clips"
    save_analysis(output / "cache", reference, [clip(0.8)], voice(0, 1))
    call_clips = [clip(0.8 + 3 * i) for i in range(4)] + [clip(12.8, confidence=0.4)]
    save_analysis(output / "cache", call, call_clips, np.concatenate([voice(1, 2), voice(0, 3, seed=5)]))
    save_manifest(output, [reference], [call])
    rebuild(output, monkeypatch)
    return output


def kept(output) -> list[str]:
    with open(output / "metadata.csv", encoding="utf-8") as metadata:
        return [row[0] for row in csv.reader(metadata, delimiter="|")]


def test_only_usable_clips_wait_to_be_marked_and_the_reference_is_kept(videos):
    intro, call = load_sources(videos)
    names = [c.clip for c in load_candidates(videos)]
    assert names == [intro.clip_name(0)] + [call.clip_name(i) for i in range(4)]  # not the unsure transcript
    assert sorted(p.name for p in (videos / "candidates").iterdir()) == sorted(names)
    assert kept(videos) == [intro.clip_name(0)]  # nothing in the call is marked yet


def test_rebuild_keeps_only_clips_marked_as_mine(videos, monkeypatch):
    _, call = load_sources(videos)
    candidates = {c.clip: c for c in load_candidates(videos)}
    for index, label in [(0, NOT_MINE), (2, MINE), (3, UNSURE)]:
        save_label(videos, candidates[call.clip_name(index)], label)
    rebuild(videos, monkeypatch)

    assert kept(videos)[1:] == [call.clip_name(2)]
    assert sorted(p.name for p in (videos / "wav").iterdir()) == sorted(kept(videos))
    with zipfile.ZipFile(videos / "voice_dataset.zip") as archive:
        assert sorted(archive.namelist()) == ["metadata.csv"] + sorted(f"wav/{name}" for name in kept(videos))
    with open(videos / "review.csv", encoding="utf-8") as review:
        reasons = {row["clip"]: row["reasons"] for row in csv.DictReader(review)}
    assert reasons[call.clip_name(0)] == NOT_MINE
    assert reasons[call.clip_name(1)] == "not marked yet"
    assert reasons[call.clip_name(4)] == "unsure transcript"


def test_an_answer_is_dropped_when_its_clip_was_cut_again(videos):
    candidates = load_candidates(videos)
    save_label(videos, candidates[1], MINE)
    recut = [dataclasses.replace(candidates[1], text="Somebody else's words.")]
    assert load_labels(videos, candidates) == {candidates[1].clip: MINE}
    assert load_labels(videos, recut) == {}


def test_page_plays_the_likeliest_clips_first_and_saves_each_answer(videos, monkeypatch):
    monkeypatch.setattr("research_analyst.config.settings", dataclasses.replace(settings, voice_clips_dir=videos))
    app = AppTest.from_file(str(SCRIPTS / "review_voice_clips.py"), default_timeout=60).run()
    assert not app.exception
    _, call = load_sources(videos)

    def click(label: str) -> None:
        next(b for b in app.button if b.label == label).click().run()
        assert not app.exception

    def showing() -> str:
        return app.caption[1].value  # under the clip: its source, time and position

    at = {"0:06": call.clip_name(2), "0:09": call.clip_name(3)}  # voice 0's clips, like the reference
    order = []
    for answer, position in [("Mine", 1), ("Not mine", 2)]:
        assert f"clip {position} of 4" in showing()
        order.append(next(clip for time, clip in at.items() if f"call.wav at {time}" in showing()))
        click(answer)
    click("Back to the previous clip")
    assert "you said: not mine" in showing()

    saved = json.loads((videos / "labels.json").read_text(encoding="utf-8"))
    assert {clip: entry["label"] for clip, entry in saved.items()} == {order[0]: MINE, order[1]: NOT_MINE}
    assert set(order) == set(at.values())
