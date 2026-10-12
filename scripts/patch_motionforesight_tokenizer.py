#!/usr/bin/env python3
"""Add an explicit offline tokenizer override to the MotionForesight runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)"""

NEW = """        # Initialize tokenizer. The release asset downloader does not include
        # google/umt5-xxl, so allow an audited local tokenizer directory offline.
        tokenizer_override = os.environ.get(\"MOTIONFORESIGHT_TOKENIZER_DIR\")
        if tokenizer_override:
            tokenizer_config = ModelConfig(path=tokenizer_override, skip_download=True)
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8")
    if NEW in text:
        print(args.path)
        return
    if text.count(OLD) != 1:
        raise SystemExit("tokenizer initialization block not found")
    args.path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(args.path)


if __name__ == "__main__":
    main()
