import csv
import dataclasses
import io
import wave

import numpy as np
from streamlit.testing.v1 import AppTest

from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.voice_dataset import (
    SAMPLE_RATE,
    check_clip,
    load_audio,
    load_prompts,
    recorded_indices,
    save_clip,
    split_utterances,
    to_wav,
    trim_silence,
)

RECORDER = str(PROJECT_ROOT / "scripts" / "record_voice.py")

TEXT = "The two men shook hands for the twentieth time."  # 47 characters, about 3.4 s of speech


def speech(seconds: float = 3.4, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def with_pauses(clip: np.ndarray, noise: float = 0.0005) -> np.ndarray:
    """Speech with a pause either side (quiet room tone), which is what the noise check measures."""
    rng = np.random.default_rng(0)
    full = np.concatenate([silence(0.5), clip, silence(0.5)])
    return full + rng.normal(0, noise, len(full)).astype(np.float32)


def test_prompts_file_is_usable():
    prompts = load_prompts(PROJECT_ROOT / "datasets" / "voice_prompts.txt")
    assert len(prompts) == 593
    assert not any("|" in prompt or prompt.startswith("#") for prompt in prompts)


def test_trim_silence_keeps_speech_and_short_padding():
    trimmed = trim_silence(np.concatenate([silence(2), speech(1), silence(2)]))
    assert 1.0 <= len(trimmed) / SAMPLE_RATE <= 1.4


def test_trim_silence_leaves_empty_audio_alone():
    assert len(trim_silence(silence(1))) == SAMPLE_RATE


def test_good_clip_has_no_issues():
    assert check_clip(with_pauses(speech()), TEXT) == []


def test_check_clip_flags_each_problem():
    assert "No sound" in check_clip(silence(3), TEXT)[0]
    assert any("Clipping" in i for i in check_clip(with_pauses(speech(amplitude=3).clip(-1, 1)), TEXT))
    assert any("quiet" in i for i in check_clip(with_pauses(speech(amplitude=0.05), noise=0.00005), TEXT))
    assert any("Noisy" in i for i in check_clip(with_pauses(speech(), noise=0.1), TEXT))
    assert any("cut off" in i for i in check_clip(with_pauses(speech(0.5)), TEXT))
    assert any("Too long" in i for i in check_clip(with_pauses(speech(30)), TEXT))


def test_save_clip_writes_wav_and_metadata_in_piper_format(tmp_path):
    prompts = ["First sentence.", "Second sentence.", "Third sentence."]
    save_clip(tmp_path, prompts, 2, speech(1))
    save_clip(tmp_path, prompts, 0, speech(1))

    with open(tmp_path / "metadata.csv", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file, delimiter="|"))
    assert rows == [["clip_0001.wav", "First sentence."], ["clip_0003.wav", "Third sentence."]]
    assert recorded_indices(tmp_path, prompts) == {0, 2}
    with wave.open(str(tmp_path / "wav" / "clip_0001.wav"), "rb") as wav_file:
        assert (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) == (1, 2, SAMPLE_RATE)
    assert not list(tmp_path.rglob("*.tmp"))


def test_save_clip_replaces_an_earlier_take(tmp_path):
    prompts = ["Only sentence."]
    save_clip(tmp_path, prompts, 0, speech(1))
    save_clip(tmp_path, prompts, 0, speech(2))
    with wave.open(str(tmp_path / "wav" / "clip_0001.wav"), "rb") as wav_file:
        assert wav_file.getnframes() == SAMPLE_RATE * 2
    assert len((tmp_path / "metadata.csv").read_text(encoding="utf-8").splitlines()) == 1


def timed(text: str, start: float, step: float = 0.4, probability: float = 0.9) -> list[tuple]:
    """Words of `text` spoken back to back from `start`, as (start, end, word, probability)."""
    return [(start + i * step, start + (i + 1) * step - 0.05, f" {w}", probability) for i, w in enumerate(text.split())]


def test_split_utterances_breaks_at_sentence_ends_and_pauses():
    words = (
        timed("We got the startup credits.", 0.0)  # ends 1.95
        + timed("Now marketing is next", 2.1)  # no full stop, ends 3.65
        + timed("Thanks a lot everyone.", 5.0)  # after a 1.35 s pause
    )
    clips = split_utterances(words)
    assert [c.text for c in clips] == ["We got the startup credits.", "Now marketing is next", "Thanks a lot everyone."]
    # Cuts stop halfway to the neighbouring words, and keep the full lead-in and tail where there's room.
    assert clips[0].start == 0.0 and abs(clips[0].end - (1.95 + 2.1) / 2) < 1e-9
    assert abs(clips[1].end - (3.65 + 0.3)) < 1e-9 and abs(clips[2].start - (5.0 - 0.25)) < 1e-9
    assert all(abs(c.confidence - 0.9) < 1e-9 for c in clips)


def test_split_utterances_drops_clips_of_untrainable_length():
    words = (
        timed("Yeah.", 0.0)  # too few words
        + timed("Okay sure fine.", 2.0, step=0.2)  # 0.55 s, too short
        + timed(" ".join(["word"] * 40) + ".", 4.0)  # 16 s with no break, too long
        + timed("This one is fine.", 22.0)
    )
    assert [c.text for c in split_utterances(words)] == ["This one is fine."]


def test_split_utterances_reports_least_certain_word_and_strips_separators():
    words = timed("Send me a message", 0.0) + [(1.6, 1.9, " now|.", 0.1)]
    (clip,) = split_utterances(words)
    assert clip.text == "Send me a message now ." and clip.min_confidence == 0.1


def test_load_audio_reads_and_resamples(tmp_path):
    to_wav(speech(1), tmp_path / "tone.wav")
    assert abs(len(load_audio(tmp_path / "tone.wav", 16000)) - 16000) < 50


def click(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def use_dataset_dir(monkeypatch, path) -> None:
    monkeypatch.setattr("research_analyst.config.settings", dataclasses.replace(settings, voice_dataset_dir=path))


def test_recorder_page_navigates(tmp_path, monkeypatch):
    use_dataset_dir(monkeypatch, tmp_path)
    app = AppTest.from_file(RECORDER, default_timeout=60).run()
    assert not app.exception
    assert app.session_state["index"] == 0
    click(app, "Skip this sentence")
    assert app.session_state["index"] == 1
    click(app, "Previous")
    assert app.session_state["index"] == 0


def test_recorder_page_saves_a_take_and_resumes_after_it(tmp_path, monkeypatch):
    use_dataset_dir(monkeypatch, tmp_path)
    take = io.BytesIO()
    to_wav(np.concatenate([silence(1), speech(4), silence(1)]), take)
    monkeypatch.setattr("streamlit.audio_input", lambda *args, **kwargs: io.BytesIO(take.getvalue()))

    app = AppTest.from_file(RECORDER, default_timeout=60).run()
    assert not app.exception
    click(app, "Save and next")

    assert app.session_state["index"] == 1
    with wave.open(str(tmp_path / "wav" / "clip_0001.wav"), "rb") as wav_file:
        assert wav_file.getnframes() / SAMPLE_RATE < 4.5  # the surrounding silence was trimmed
    assert (tmp_path / "metadata.csv").read_text(encoding="utf-8").startswith("clip_0001.wav|Author of the danger")

    reopened = AppTest.from_file(RECORDER, default_timeout=60).run()
    assert reopened.session_state["index"] == 1
