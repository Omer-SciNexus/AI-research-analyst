"""Deploy the API + web UI to a Hugging Face Space (Docker SDK, free CPU tier).

The image is built on Hugging Face's servers, so only the source code (~100 KB)
is uploaded from this machine.

Needs:
    HF_TOKEN       a Hugging Face access token with write permission
    GROQ_API_KEY   free key from https://console.groq.com/keys (stored as a Space secret)

Usage:
    uv run python scripts/deploy_space.py --space <your-username>/research-analyst [--private]
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "README.md", "scripts/warmup.py"]
FOLDERS = ["src", "web", "samples"]

SPACE_CARD = """---
title: Research Analyst
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Cited RAG chat and a LangGraph research agent
---

"""


def stage(target: Path) -> None:
    for name in FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target / name)
    for folder in FOLDERS:
        shutil.copytree(ROOT / folder, target / folder, ignore=shutil.ignore_patterns("__pycache__"))
    readme = target / "README.md"
    readme.write_text(SPACE_CARD + readme.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token, groq_key = os.getenv("HF_TOKEN"), os.getenv("GROQ_API_KEY")
    if not token or not groq_key:
        sys.exit("Set HF_TOKEN and GROQ_API_KEY first (see the docstring).")

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_secret(args.space, "GROQ_API_KEY", groq_key)

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space", commit_message="Deploy Research Analyst")

    print(f"Deployed. Build logs and app: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
