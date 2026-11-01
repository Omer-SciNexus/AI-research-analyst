"""Voice: local speech-to-text and text-to-speech for Document Chat.

STT is faster-whisper (CTranslate2, CPU, int8) - spoken questions (Korean by
default) are translated to English text before going through the normal RAG
pipeline. TTS is Piper
(ONNX, CPU) - answers are synthesized to a WAV clip for playback. Both
models run fully offline once downloaded.
"""

import io
import wave
from functools import lru_cache
from typing import BinaryIO

import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice

from research_analyst.config import settings


@lru_cache
def get_whisper() -> WhisperModel:
    # Prefer a pre-downloaded copy (scripts/download_voice_models.sh) over the hub's auto-download.
    local = settings.models_dir / f"faster-whisper-{settings.stt_model}"
    model = str(local) if (local / "model.bin").exists() else settings.stt_model
    return WhisperModel(model, device="cpu", compute_type="int8", download_root=str(settings.models_dir))


def load_wav(audio: BinaryIO, rate: int = 16000) -> np.ndarray:
    """16-bit PCM WAV -> mono float32 at `rate` Hz, the input Whisper expects."""
    with wave.open(audio, "rb") as wav_file:
        source_rate, channels = wav_file.getframerate(), wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != rate:
        positions = np.arange(0, len(samples), source_rate / rate)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def transcribe(audio: BinaryIO) -> str:
    """Speech in `settings.stt_language` -> English text (Whisper's translate task)."""
    # Decoded here rather than by faster-whisper: its PyAV decoder breaks on av>=16.
    segments, _info = get_whisper().transcribe(load_wav(audio), task="translate", language=settings.stt_language)
    return " ".join(segment.text.strip() for segment in segments).strip()


@lru_cache
def get_piper() -> PiperVoice:
    model_path = settings.models_dir / f"{settings.tts_voice}.onnx"
    return PiperVoice.load(model_path)


def synthesize(text: str) -> bytes:
    """Render `text` to a WAV clip (bytes), ready for st.audio()."""
    chunks = list(get_piper().synthesize(text))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(chunks[0].sample_channels if chunks else 1)
        wav_file.setsampwidth(chunks[0].sample_width if chunks else 2)
        wav_file.setframerate(chunks[0].sample_rate if chunks else 22050)
        for chunk in chunks:
            wav_file.writeframes(chunk.audio_int16_bytes)
    return buffer.getvalue()
