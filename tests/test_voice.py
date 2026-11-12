import dataclasses

from research_analyst import voice
from research_analyst.config import settings


def use(monkeypatch, tmp_path, name: str) -> None:
    monkeypatch.setattr(voice, "settings", dataclasses.replace(settings, models_dir=tmp_path, tts_voice=name))


def test_trained_voice_is_used_once_both_files_exist(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path, "en_US-omer-medium")
    (tmp_path / "en_US-omer-medium.onnx").write_bytes(b"")
    (tmp_path / "en_US-omer-medium.onnx.json").write_text("{}")
    assert voice.tts_voice() == "en_US-omer-medium"


def test_falls_back_to_default_voice_while_files_are_missing(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path, "en_US-omer-medium")
    assert voice.tts_voice() == voice.DEFAULT_TTS_VOICE
    (tmp_path / "en_US-omer-medium.onnx").write_bytes(b"")  # config .json still missing
    assert voice.tts_voice() == voice.DEFAULT_TTS_VOICE
