"""Record yourself reading prompts, to build a dataset for training a Piper voice.

Progress is saved after every clip, so you can stop and come back at any time.

Usage:
    uv run streamlit run scripts/record_voice.py
Output: data/voice_dataset/ (metadata.csv + wav/), kept out of git and the Docker image
"""

import io

import streamlit as st

from research_analyst.config import PROJECT_ROOT, settings
from research_analyst.voice import load_wav
from research_analyst.voice_dataset import (
    SAMPLE_RATE,
    TARGET_CLIPS,
    check_clip,
    clip_name,
    load_prompts,
    recorded_indices,
    save_clip,
    to_wav,
    trim_silence,
)

DATASET_DIR = settings.voice_dataset_dir
PROMPTS = load_prompts(PROJECT_ROOT / "datasets" / "voice_prompts.txt")


def next_missing(recorded: set[int], start: int = 0) -> int:
    """First prompt without a clip at or after `start`, or `start` itself if there is none."""
    return next((i for i in range(start, len(PROMPTS)) if i not in recorded), start)


def go_to(index: int) -> None:
    st.session_state.index = max(0, min(index, len(PROMPTS) - 1))


st.set_page_config(page_title="Record Voice", page_icon="🎙️")
st.title("Record your voice")

recorded = recorded_indices(DATASET_DIR, PROMPTS)
if "index" not in st.session_state:
    st.session_state.index = next_missing(recorded)
index = st.session_state.index

st.progress(min(len(recorded) / TARGET_CLIPS, 1.0), f"{len(recorded)} of {TARGET_CLIPS} clips recorded")
if len(recorded) >= TARGET_CLIPS:
    st.success("You have enough clips to train. More clips still improve the voice.")

with st.expander("How to get a good recording"):
    st.markdown(
        "- Use the same room, microphone and distance in every session.\n"
        "- Read in the calm, even voice you want the answers spoken in.\n"
        "- Pause about half a second after pressing record and before stopping.\n"
        "- Read the sentence exactly as written. If you stumble, record it again."
    )

st.caption(f"Sentence {index + 1} of {len(PROMPTS)}")
st.subheader(PROMPTS[index])

existing = DATASET_DIR / "wav" / clip_name(index)
if index in recorded:
    st.info("Already recorded. Listen below, or record again to replace it.")
    st.audio(existing.read_bytes(), format="audio/wav")

# The key changes with the sentence, so the recorder starts empty each time.
audio = st.audio_input("Record this sentence", sample_rate=SAMPLE_RATE, key=f"mic-{index}")
if audio is not None:
    clip = trim_silence(load_wav(audio, SAMPLE_RATE))
    issues = check_clip(clip, PROMPTS[index])
    for issue in issues:
        st.warning(issue)
    if not issues:
        st.success("Levels look good. Listen, then save.")
    preview = io.BytesIO()
    to_wav(clip, preview)
    st.audio(preview.getvalue(), format="audio/wav")
    if st.button("Save and next", type="primary", width="stretch"):
        save_clip(DATASET_DIR, PROMPTS, index, clip)
        recorded.add(index)
        go_to(next_missing(recorded, index + 1))
        st.rerun()

back, skip = st.columns(2)
back.button("Previous", width="stretch", disabled=index == 0, on_click=go_to, args=(index - 1,))
skip.button("Skip this sentence", width="stretch", on_click=go_to, args=(index + 1,))
