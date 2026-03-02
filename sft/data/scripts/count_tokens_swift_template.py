#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from swift.llm import get_model_tokenizer, get_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count tokens in a JSONL dataset using ms-swift chat template "
            "(qwen3_nothinking + qwen_en by default)."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset-extended/dataset_shuffled.jsonl"
        ),
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/wbl-fast/usrs/ethan/small-agent/sft/checkpoints/LocoreMind-LocoOperator-4B",
        help="Model/tokenizer path used for training.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen3_nothinking",
        help="ms-swift template name.",
    )
    parser.add_argument(
        "--agent-template",
        type=str,
        default="qwen_en",
        help="ms-swift agent template name.",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=5.0,
        help="Progress print interval in seconds.",
    )
    return parser.parse_args()


def parse_tools_field(raw_tools: Any) -> Any:
    if not isinstance(raw_tools, str):
        return raw_tools
    stripped = raw_tools.strip()
    if not stripped:
        return raw_tools
    if stripped[0] not in "[{":
        return raw_tools
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw_tools


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    os.environ.setdefault(
        "MODELSCOPE_CACHE",
        "/wbl-fast/usrs/ethan/small-agent/sft/.modelscope_cache",
    )
    os.environ.setdefault(
        "HF_HOME",
        "/wbl-fast/usrs/ethan/small-agent/sft/.hf_home",
    )
    Path(os.environ["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    start = time.time()
    last_log = 0.0
    total_tokens = 0
    rows = 0
    errors = 0
    file_size = dataset_path.stat().st_size

    _, tokenizer = get_model_tokenizer(args.model_path, load_model=False)
    template = get_template(
        args.template, tokenizer, agent_template=args.agent_template
    )

    print(
        f"[start] dataset={dataset_path} size_bytes={file_size} "
        f"template={args.template} agent_template={args.agent_template}",
        flush=True,
    )

    bytes_read = 0
    with dataset_path.open("rb") as f:
        for raw_line in f:
            bytes_read += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line)
                tools = parse_tools_field(row.get("tools"))
                encoded = template.encode(
                    {"messages": row.get("messages", []), "tools": tools}
                )
                total_tokens += len(encoded.get("input_ids", []))
            except Exception:
                errors += 1
            rows += 1

            now = time.time()
            if now - last_log >= args.log_interval_seconds:
                elapsed = now - start
                progress = min(1.0, (bytes_read / file_size) if file_size else 0.0)
                eta_seconds = (
                    (elapsed / progress - elapsed) if progress > 0 else float("inf")
                )
                est_total_tokens = (
                    int(total_tokens / progress) if progress > 0 else total_tokens
                )
                tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    "[progress] "
                    f"{progress * 100:6.2f}% "
                    f"rows={rows:,} "
                    f"tokens={total_tokens:,} "
                    f"tok/s={tok_per_sec:,.1f} "
                    f"eta={format_duration(eta_seconds)} "
                    f"est_total_tokens={est_total_tokens:,} "
                    f"errors={errors:,}",
                    flush=True,
                )
                last_log = now

    elapsed = time.time() - start
    avg_tokens_per_row = (total_tokens / rows) if rows else 0.0
    print(
        "[done] "
        f"100.00% rows={rows:,} tokens={total_tokens:,} "
        f"avg_tokens_per_row={avg_tokens_per_row:,.2f} "
        f"errors={errors:,} elapsed={format_duration(elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()


# Rows processed: 255,749
# Final token count: 2,322,408,315
# Average tokens/row: 9,080.81
# Errors: 0
# Total elapsed: 01:13:19
