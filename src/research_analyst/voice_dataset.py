"""Build a Piper training dataset from clips of one speaker: read prompts, or sentences cut from videos.

Layout (what `python -m piper.train` reads as --data.csv_path / --data.audio_dir):
    <dataset_dir>/metadata.csv      clip_0001.wav|Text that was read.
    <dataset_dir>/wav/clip_0001.wav 16-bit mono at SAMPLE_RATE
"""

import os
import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 22050  # Piper "medium" quality, the rate of the checkpoints it can fine-tune from
TARGET_CLIPS = 300

# Sentence clips cut from longer speech
_MAX_PAUSE = 0.5  # seconds between words that ends a clip (often a change of speaker)
_LEAD_IN, _TAIL = 0.25, 0.3  # seconds kept around the words, never past halfway to a neighbouring word
MIN_SECONDS, MAX_SECONDS, MIN_WORDS = 1.0, 12.0, 3

_FRAME_SECONDS = 0.02
_SILENCE_DB = -35.0  # relative to the loudest frame; low enough to keep quiet word endings
_PADDING_SECONDS = 0.15
_MIN_SNR_DB = 25.0
_CHARS_PER_SECOND = (6.0, 24.0)  # conversational English is around 14


@dataclass
class Utterance:
    start: float  # cut points in seconds, including a short lead-in and tail
    end: float
    text: str
    confidence: float  # mean Whisper word probability
    min_confidence: float  # least certain word


def load_audio(path: Path, rate: int) -> np.ndarray:
    """Any audio or video file -> mono float32 at `rate` Hz."""
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        chunks = []
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().ravel() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().ravel() for f in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, dtype=np.float32)


def split_utterances(words: list[tuple[float, float, str, float]]) -> list[Utterance]:
    """Group timed words (start, end, text, probability) into sentence clips of a trainable length.

    A clip ends at a sentence end or a pause. Clips that are too short, too long or too few
    words are dropped rather than cut mid-sentence.
    """
    groups, current = [], []
    for index, word in enumerate(words):
        if current and word[0] - words[current[-1]][1] > _MAX_PAUSE:
            groups.append(current)
            current = []
        current.append(index)
        if word[2].rstrip().endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    utterances = []
    for group in groups:
        first, last = words[group[0]], words[group[-1]]
        if len(group) < MIN_WORDS or not MIN_SECONDS <= last[1] - first[0] <= MAX_SECONDS:
            continue
        start = first[0] - _LEAD_IN
        if group[0] > 0:
            start = max(start, (words[group[0] - 1][1] + first[0]) / 2)
        end = last[1] + _TAIL
        if group[-1] + 1 < len(words):
            end = min(end, (last[1] + words[group[-1] + 1][0]) / 2)
        probabilities = [words[i][3] for i in group]
        text = " ".join("".join(words[i][2] for i in group).replace("|", " ").split())
        utterances.append(Utterance(max(0.0, start), end, text, float(np.mean(probabilities)), float(min(probabilities))))
    return utterances


def load_prompts(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def clip_name(index: int) -> str:
    return f"clip_{index + 1:04d}.wav"


def _frame_levels(samples: np.ndarray, rate: int) -> np.ndarray:
    size = int(rate * _FRAME_SECONDS)
    count = len(samples) // size
    if count == 0:
        return np.zeros(0)
    frames = samples[: count * size].reshape(count, size)
    return np.sqrt(np.mean(frames**2, axis=1))


def trim_silence(samples: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cut the dead air (and mic-button clicks in it) around the speech, keeping a short pad."""
    levels = _frame_levels(samples, rate)
    if levels.size == 0 or levels.max() == 0:
        return samples
    voiced = np.flatnonzero(levels >= levels.max() * 10 ** (_SILENCE_DB / 20))
    frame = int(rate * _FRAME_SECONDS)
    padding = int(rate * _PADDING_SECONDS)
    return samples[max(0, voiced[0] * frame - padding) : (voiced[-1] + 1) * frame + padding]


def check_clip(samples: np.ndarray, text: str, rate: int = SAMPLE_RATE) -> list[str]:
    """Problems likely to hurt training, as messages for the speaker. Empty means the clip looks fine."""
    levels = _frame_levels(samples, rate)
    if levels.size == 0 or levels.max() < 1e-4:
        return ["No sound was captured. Check the microphone."]

    issues = []
    peak = float(np.abs(samples).max())
    if np.mean(np.abs(samples) >= 0.99) > 0.001:
        issues.append("Clipping: the recording is distorted. Move back from the microphone.")
    elif peak < 0.1:
        issues.append("Very quiet. Move closer or raise the microphone level.")

    noise, speech = np.percentile(levels, 10), np.percentile(levels, 95)
    if noise > 0 and 20 * np.log10(speech / noise) < _MIN_SNR_DB:
        issues.append("Noisy background: your speech doesn't stand out from it. Try a quieter room.")

    speed = len(text) / (len(samples) / rate)
    if speed > _CHARS_PER_SECOND[1]:
        issues.append("Too short for this sentence. Was the recording cut off?")
    elif speed < _CHARS_PER_SECOND[0]:
        issues.append("Too long for this sentence. Avoid long pauses, and read it once.")
    return issues


def to_wav(samples: np.ndarray, target, rate: int = SAMPLE_RATE) -> None:
    """Write float samples in [-1, 1] as 16-bit mono PCM to a path or binary file object."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(target) if isinstance(target, Path) else target, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm.tobytes())


def recorded_indices(dataset_dir: Path, prompts: list[str]) -> set[int]:
    return {i for i in range(len(prompts)) if (dataset_dir / "wav" / clip_name(i)).exists()}


def write_metadata(dataset_dir: Path, prompts: list[str]) -> None:
    """Rebuild metadata.csv from the clips on disk, so the two can't disagree."""
    lines = [f"{clip_name(i)}|{prompts[i]}\n" for i in sorted(recorded_indices(dataset_dir, prompts))]
    temporary = dataset_dir / "metadata.csv.tmp"
    temporary.write_text("".join(lines), encoding="utf-8", newline="\n")
    os.replace(temporary, dataset_dir / "metadata.csv")


def save_clip(dataset_dir: Path, prompts: list[str], index: int, samples: np.ndarray) -> Path:
    """Store (or replace) the take for prompt `index`, without leaving a half-written WAV behind."""
    (dataset_dir / "wav").mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "wav" / clip_name(index)
    temporary = path.with_suffix(".tmp")
    to_wav(samples, temporary)
    os.replace(temporary, path)
    write_metadata(dataset_dir, prompts)
    return path
