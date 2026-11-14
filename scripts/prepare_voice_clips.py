"""Cut sentence clips of one speaker out of videos, to train a Piper voice.

Pass 1 (slow, cached per video): Whisper transcribes with word timings, the words are
grouped into sentence clips, and each clip gets a speaker embedding.
Pass 2 (fast): clips with a clear transcript and clean audio are written to candidates/;
those from the reference videos, and those the speaker marked as theirs in
scripts/review_voice_clips.py, go into the dataset in Piper's format.

Usage:
    uv sync --group voice-prep
    uv run python scripts/prepare_voice_clips.py --reference intro.mp4 --sources call1.mp4 call2.mp4
    uv run streamlit run scripts/review_voice_clips.py   # listen and mark the clips that are you
    uv run python scripts/prepare_voice_clips.py         # rebuild the dataset from your answers
Output: data/voice_clips/ (metadata.csv, wav/, voice_dataset.zip of both, review.csv), kept out
of git and the Docker image
Needs: sherpa-onnx and data/models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
"""

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np

from research_analyst.config import settings
from research_analyst.voice import get_whisper
from research_analyst.voice_clips import (
    Candidate,
    confident,
    is_mine,
    load_analysis,
    load_labels,
    load_sources,
    reference_profile,
    save_analysis,
    save_candidates,
    save_manifest,
)
from research_analyst.voice_dataset import SAMPLE_RATE, check_clip, load_audio, split_utterances, to_wav, trim_silence

SPEAKER_RATE = 16000  # what Whisper and the speaker model take
SPEAKER_MODEL = settings.models_dir / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


class SpeakerEmbedder:
    def __init__(self, model: Path):
        import sherpa_onnx  # voice-prep group; only needed to analyse new videos, not to rebuild

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=4)
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self.extractor.dim

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SPEAKER_RATE, waveform=samples)
        stream.input_finished()
        vector = np.array(self.extractor.compute(stream), dtype=np.float32)
        return vector / np.linalg.norm(vector)


def analyze(path: Path, cache_dir: Path, embed: SpeakerEmbedder) -> None:
    """Pass 1 for one video, skipped when it is already cached."""
    if load_analysis(cache_dir, path) is not None:
        return
    print(f"transcribing {path.name} (slow on CPU)", flush=True)
    samples = load_audio(path, SPEAKER_RATE)
    segments, _info = get_whisper().transcribe(
        samples,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=False,  # stops one mistake repeating through a long recording
    )
    words = [(w.start, w.end, w.word, w.probability) for segment in segments for w in (segment.words or [])]
    utterances = split_utterances(words)
    rows = [embed(samples[int(u.start * SPEAKER_RATE) : int(u.end * SPEAKER_RATE)]) for u in utterances]
    save_analysis(cache_dir, path, utterances, np.stack(rows) if rows else np.empty((0, embed.dim), np.float32))
    print(f"  {len(utterances)} sentence clips", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, nargs="+", help="videos with only the target speaker")
    parser.add_argument("--sources", type=Path, nargs="*", default=[], help="videos that also contain other voices")
    args = parser.parse_args()

    output = settings.voice_clips_dir
    if args.reference:
        embed = SpeakerEmbedder(SPEAKER_MODEL)
        for video in dict.fromkeys(args.reference + args.sources):
            analyze(video, output / "cache", embed)
        save_manifest(output, args.reference, args.sources)
    elif args.sources:
        parser.error("--sources needs --reference")

    sources = load_sources(output)
    if not any(s.is_reference for s in sources):
        raise SystemExit("No analysed reference video yet: pass --reference (and --sources).")
    profile = reference_profile(sources)

    candidate_dir, wav_dir = output / "candidates", output / "wav"
    for folder in (candidate_dir, wav_dir):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.wav"):  # from an earlier run
            old.unlink()

    rows, candidates = [], []
    for source in sources:
        print(f"{source.path.name}: {len(source.utterances)} clips", flush=True)
        audio = load_audio(source.path, SAMPLE_RATE)
        for index, (utterance, embedding) in enumerate(zip(source.utterances, source.embeddings)):
            name = source.clip_name(index)
            reasons = [] if confident(utterance) else ["unsure transcript"]
            if not reasons:
                clip = trim_silence(audio[int(utterance.start * SAMPLE_RATE) : int(utterance.end * SAMPLE_RATE)])
                reasons = check_clip(clip, utterance.text)
                if not reasons:
                    to_wav(clip, candidate_dir / name)
                    candidates.append(Candidate(
                        name, source.path.name, source.is_reference, round(utterance.start, 2),
                        round(len(clip) / SAMPLE_RATE, 2), round(float(embedding @ profile), 3), utterance.text,
                    ))
            rows.append({
                "clip": name, "reasons": reasons, "confidence": round(utterance.confidence, 3),
                "start": round(utterance.start, 2), "end": round(utterance.end, 2),
                "source": source.path.name, "text": utterance.text,
            })
    save_candidates(output, candidates)

    labels = load_labels(output, candidates)
    kept = {c.clip for c in candidates if is_mine(c, labels)}
    for name in kept:
        shutil.copyfile(candidate_dir / name, wav_dir / name)
    for row in rows:
        if not row["reasons"] and row["clip"] not in kept:
            row["reasons"] = [labels.get(row["clip"], "not marked yet")]
        row["status"] = "keep" if row["clip"] in kept else "drop"
        row["reasons"] = "; ".join(row["reasons"])

    fields = ["clip", "status", "reasons", "confidence", "start", "end", "source", "text"]
    with open(output / "review.csv", "w", encoding="utf-8", newline="") as review:
        writer = csv.DictWriter(review, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "metadata.csv", "w", encoding="utf-8", newline="\n") as metadata:
        metadata.writelines(f"{r['clip']}|{r['text']}\n" for r in rows if r["status"] == "keep")
    # One file to upload for training (notebooks/train_piper_voice.ipynb); WAV barely compresses.
    with zipfile.ZipFile(output / "voice_dataset.zip", "w", zipfile.ZIP_STORED) as archive:
        archive.write(output / "metadata.csv", "metadata.csv")
        for name in sorted(kept):
            archive.write(wav_dir / name, f"wav/{name}")

    to_mark = [c for c in candidates if not c.reference]
    minutes = sum(c.seconds for c in candidates if c.clip in kept) / 60
    print(f"\n{len(candidates)} usable clips; {sum(c.clip in labels for c in to_mark)} of {len(to_mark)} marked")
    print(f"dataset: {len(kept)} clips, {minutes:.1f} min of your speech")
    print(f"every clip's decision: {output / 'review.csv'}")


if __name__ == "__main__":
    main()
