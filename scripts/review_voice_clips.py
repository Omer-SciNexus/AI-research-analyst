"""Listen to each usable clip and mark whether it is you.

Run after scripts/prepare_voice_clips.py, then run that script again (without arguments)
to rebuild the dataset from your answers. Every answer is saved at once, so you can stop
and come back.

Usage:
    uv run streamlit run scripts/review_voice_clips.py
"""

import streamlit as st

from research_analyst.config import settings
from research_analyst.voice_clips import MINE, NOT_MINE, UNSURE, is_mine, load_candidates, load_labels, save_label

TARGET_MINUTES = 20  # enough speech to fine-tune a Piper voice well

st.set_page_config(page_title="Mark Your Clips", page_icon="🎧")
st.title("Which clips are you?")

output = settings.voice_clips_dir
candidates = load_candidates(output)
if not candidates:
    st.info("Nothing to mark yet. Run scripts/prepare_voice_clips.py first.")
    st.stop()
labels = load_labels(output, candidates)
# The clips that sound most like your reference videos come first, so the target is reached sooner.
queue = sorted((c for c in candidates if not c.reference), key=lambda c: -c.score)

minutes = sum(c.seconds for c in candidates if is_mine(c, labels)) / 60
done = sum(c.clip in labels for c in queue)
st.progress(min(minutes / TARGET_MINUTES, 1.0), text=f"Your speech: **{minutes:.1f} of {TARGET_MINUTES} min**")
st.caption(
    f"{done} of {len(queue)} clips marked. Mark a clip as yours only if every word in it is you. "
    "Once the bar is full you can stop: the rest is optional."
)

if "position" not in st.session_state:
    st.session_state.position = next((i for i, c in enumerate(queue) if c.clip not in labels), len(queue))
position = st.session_state.position


def answer(label: str) -> None:
    save_label(output, queue[position], label)
    st.session_state.position = next(
        (i for i in range(position + 1, len(queue)) if queue[i].clip not in load_labels(output, candidates)),
        len(queue),
    )


if position >= len(queue):
    st.success("All clips are marked. Now rebuild the dataset: `uv run python scripts/prepare_voice_clips.py`")
else:
    clip = queue[position]
    with st.container(border=True):
        st.audio((output / "candidates" / clip.clip).read_bytes(), format="audio/wav", autoplay=True)
        st.markdown(f"“{clip.text}”")
        earlier = labels.get(clip.clip)
        st.caption(
            f"{clip.source} at {int(clip.start // 60)}:{int(clip.start % 60):02d} · clip {position + 1} of {len(queue)}"
            + (f" · you said: {earlier}" if earlier else "")
        )
    mine, not_mine, unsure = st.columns(3)
    mine.button("Mine", type="primary", width="stretch", on_click=answer, args=(MINE,))
    not_mine.button("Not mine", width="stretch", on_click=answer, args=(NOT_MINE,))
    unsure.button("Not sure", width="stretch", on_click=answer, args=(UNSURE,))

if position > 0 and st.button("Back to the previous clip", width="stretch"):
    st.session_state.position = position - 1
    st.rerun()
