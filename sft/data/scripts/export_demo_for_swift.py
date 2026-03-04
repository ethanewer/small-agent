#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def _is_valid_messages(messages: object) -> bool:
    if not isinstance(messages, list) or not messages:
        return False

    for msg in messages:
        if not isinstance(msg, dict):
            return False

        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not role:
            return False

        if not isinstance(content, str):
            return False
    return True


def export_dataset(input_dir: Path, output_file: Path) -> tuple[int, int]:
    ds = load_from_disk(str(input_dir))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_file.open("w", encoding="utf-8") as f:
        for row in ds:
            messages = row.get("messages")
            if not _is_valid_messages(messages):
                skipped += 1
                continue

            payload = {"messages": messages}
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HF-disk demo dataset to ms-swift JSONL (messages only)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/demo-dataset/dataset"),
        help="Path to saved Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/demo-dataset/train.jsonl"),
        help="Path for output JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written, skipped = export_dataset(
        input_dir=args.input_dir, output_file=args.output_file
    )
    print(f"Wrote {written} rows to {args.output_file}")
    if skipped:
        print(f"Skipped {skipped} invalid rows")


if __name__ == "__main__":
    main()
