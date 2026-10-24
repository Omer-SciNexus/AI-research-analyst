"""Agentic research & report analyst."""

import os

# Windows without Developer Mode can't symlink; the Hugging Face cache still works, it just
# warns loudly. Set here, before any submodule imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
