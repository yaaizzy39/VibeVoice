#!/usr/bin/env python
"""Download VibeVoice-ASR-HF model files for later local/offline use."""

from __future__ import annotations

import argparse
from pathlib import Path

from transcribe_vibevoice_hf import configure_hf_cache

DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VibeVoice-ASR-HF model files.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model ID to download.")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional directory to store a full copy. If omitted, Hugging Face cache is used.",
    )
    args = parser.parse_args()

    cache_dir = configure_hf_cache()
    from huggingface_hub import snapshot_download

    kwargs = {}
    if args.local_dir:
        local_dir = Path(args.local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)

    path = snapshot_download(args.model, **kwargs)
    print(path)
    print(f"Hugging Face cache: {cache_dir}")


if __name__ == "__main__":
    main()
