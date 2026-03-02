from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson
import typer
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

app = typer.Typer(add_completion=False, help="Build Qwen SFT datasets in messages format.")

DEFAULT_DEMO_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/data/demo-dataset")
DEFAULT_FULL_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/data/full-dataset")
DEFAULT_BALANCED_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/data/balanced-dataset")
DEFAULT_ADDITIONAL_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/data/additional-data")
DEFAULT_EXTENDED_FULL_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/data/full-dataset-extended")
DEFAULT_SEED = 42
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_ESCAPED_BLOCK_RE = re.compile(r"&lt;think&gt;.*?&lt;/think&gt;", flags=re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>|&lt;think&gt;", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>|&lt;/think&gt;", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: str | None
    split: str
    messages_key: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("nvidia/Nemotron-Terminal-Corpus", "skill_based_mixed", "train", "conversations"),
    DatasetSpec("Nanbeige/ToolMind-Web-QA", "test", "train", "conversations"),
    DatasetSpec("SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K", "default", "train", "messages"),
)

NVIDIA_TERMINAL_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("nvidia/Nemotron-Terminal-Corpus", "skill_based_easy", "train", "conversations"),
    DatasetSpec("nvidia/Nemotron-Terminal-Corpus", "skill_based_medium", "train", "conversations"),
    DatasetSpec("nvidia/Nemotron-Terminal-Corpus", "skill_based_mixed", "train", "conversations"),
)

ADDITIONAL_NEMOTRON_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("nvidia/Nemotron-Agentic-v1", "default", "interactive_agent", "messages"),
    DatasetSpec("nvidia/Nemotron-Agentic-v1", "default", "tool_calling", "messages"),
    DatasetSpec("nvidia/Nemotron-Instruction-Following-Chat-v1", "default", "chat_if", "messages"),
    DatasetSpec("nvidia/Nemotron-Instruction-Following-Chat-v1", "default", "structured_outputs", "messages"),
)


def get_streaming_dataset(spec: DatasetSpec):
    if spec.name == "Nanbeige/ToolMind-Web-QA":
        # The repo contains image assets; load only JSONL data files.
        return load_dataset(
            "json",
            data_files=[
                "hf://datasets/Nanbeige/ToolMind-Web-QA/open-wiki-traj.jsonl",
                "hf://datasets/Nanbeige/ToolMind-Web-QA/syn_wikiqa.jsonl",
            ],
            split="train",
            streaming=True,
        )
    if spec.name == "nvidia/Nemotron-Agentic-v1":
        split_to_file = {
            "interactive_agent": "hf://datasets/nvidia/Nemotron-Agentic-v1/data/interactive_agent.jsonl",
            "tool_calling": "hf://datasets/nvidia/Nemotron-Agentic-v1/data/tool_calling.jsonl",
        }
        data_file = split_to_file.get(spec.split)
        if data_file is None:
            raise ValueError(f"Unsupported split for {spec.name}: {spec.split}")
        return load_dataset("json", data_files=[data_file], split="train", streaming=True)
    if spec.name == "nvidia/Nemotron-Instruction-Following-Chat-v1":
        split_to_file = {
            "chat_if": "hf://datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/data/chat_if.jsonl",
            "structured_outputs": "hf://datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/data/structured_outputs.jsonl",
        }
        data_file = split_to_file.get(spec.split)
        if data_file is None:
            raise ValueError(f"Unsupported split for {spec.name}: {spec.split}")
        return load_dataset("json", data_files=[data_file], split="train", streaming=True)
    return load_dataset(spec.name, spec.config, split=spec.split, streaming=True)


def remove_think_tags(text: str) -> str:
    cleaned = text
    while True:
        updated = THINK_BLOCK_RE.sub("", cleaned)
        updated = THINK_ESCAPED_BLOCK_RE.sub("", updated)
        if updated == cleaned:
            break
        cleaned = updated

    has_open = THINK_OPEN_RE.search(cleaned) is not None
    has_close = THINK_CLOSE_RE.search(cleaned) is not None

    # Handle malformed traces where only one side of think tags exists.
    if has_open and not has_close:
        first_open = THINK_OPEN_RE.search(cleaned)
        if first_open is not None:
            cleaned = cleaned[: first_open.start()]
    elif has_close and not has_open:
        close_matches = list(THINK_CLOSE_RE.finditer(cleaned))
        if close_matches:
            cleaned = cleaned[close_matches[-1].end() :]

    cleaned = THINK_OPEN_RE.sub("", cleaned)
    cleaned = THINK_CLOSE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if not role:
            continue
        content = remove_think_tags(to_text(msg.get("content", "")))
        normalized.append({"role": role, "content": content})

    return normalized


def normalize_messages_lightweight(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if not role:
            continue

        content_raw = msg.get("content", "")
        if isinstance(content_raw, str):
            content = remove_think_tags(content_raw)
        elif isinstance(content_raw, list):
            parts: list[str] = []
            for item in content_raw:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
            content = remove_think_tags("\n".join(parts))
        else:
            content = ""

        if not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized


def use_lightweight_normalization(spec: DatasetSpec) -> bool:
    return spec.name in {
        "nvidia/Nemotron-Agentic-v1",
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
    }


def additional_spec_jsonl_url(spec: DatasetSpec) -> str | None:
    urls = {
        ("nvidia/Nemotron-Agentic-v1", "interactive_agent"): (
            "https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/resolve/main/data/interactive_agent.jsonl?download=true"
        ),
        ("nvidia/Nemotron-Agentic-v1", "tool_calling"): (
            "https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/resolve/main/data/tool_calling.jsonl?download=true"
        ),
        ("nvidia/Nemotron-Instruction-Following-Chat-v1", "chat_if"): (
            "https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/resolve/main/data/chat_if.jsonl?download=true"
        ),
        ("nvidia/Nemotron-Instruction-Following-Chat-v1", "structured_outputs"): (
            "https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/resolve/main/data/structured_outputs.jsonl?download=true"
        ),
    }
    return urls.get((spec.name, spec.split))


def iter_rows_from_jsonl_url(spec: DatasetSpec, seed: int, limit: int | None = None) -> Iterable[dict[str, Any]]:
    url = additional_spec_jsonl_url(spec)
    if url is None:
        raise ValueError(f"No JSONL URL mapping for dataset spec: {spec.name}/{spec.split}")

    normalize = normalize_messages_lightweight if use_lightweight_normalization(spec) else normalize_messages
    accepted = 0
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        for row_id, line in enumerate(response.iter_lines()):
            if not line:
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                continue
            messages = normalize(row.get(spec.messages_key))
            if not messages:
                continue
            source_config = spec.config or "default"
            yield {
                "source_dataset": spec.name,
                "source_config": source_config,
                "source_split": spec.split,
                "row_id": f"{spec.name}:{source_config}:{spec.split}:{row_id}",
                "shuffle_key": _shuffle_key(seed, spec.name, source_config, spec.split, row_id),
                "messages": messages,
            }
            accepted += 1
            if limit is not None and accepted >= limit:
                break


def _shuffle_key(seed: int, source_dataset: str, source_config: str, source_split: str, row_id: int) -> str:
    payload = f"{seed}:{source_dataset}:{source_config}:{source_split}:{row_id}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def iter_rows(spec: DatasetSpec, seed: int, limit: int | None = None) -> Iterable[dict[str, Any]]:
    direct_jsonl_url = additional_spec_jsonl_url(spec)
    if direct_jsonl_url is not None:
        yield from iter_rows_from_jsonl_url(spec, seed=seed, limit=limit)
        return

    ds = get_streaming_dataset(spec)
    normalize = normalize_messages_lightweight if use_lightweight_normalization(spec) else normalize_messages
    accepted = 0
    for row_id, row in enumerate(ds):
        messages = normalize(row.get(spec.messages_key))
        if not messages:
            continue
        source_config = spec.config or "default"
        yield {
            "source_dataset": spec.name,
            "source_config": source_config,
            "source_split": spec.split,
            "row_id": f"{spec.name}:{source_config}:{spec.split}:{row_id}",
            "shuffle_key": _shuffle_key(seed, spec.name, source_config, spec.split, row_id),
            "messages": messages,
        }
        accepted += 1
        if limit is not None and accepted >= limit:
            break


def render_examples(rows: list[dict[str, Any]], max_examples: int = 12) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    examples: list[dict[str, Any]] = []
    for row in rows:
        if len(examples) >= max_examples:
            break
        try:
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            continue
        examples.append(
            {
                "source_dataset": row["source_dataset"],
                "source_config": row["source_config"],
                "row_id": row["row_id"],
                "rendered_text": rendered,
            }
        )
    return examples


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")


def row_messages_bytes(row: dict[str, Any]) -> int:
    return len(orjson.dumps(row["messages"]))


def total_messages_bytes(rows: list[dict[str, Any]]) -> int:
    return sum(row_messages_bytes(row) for row in rows)


def build_demo(output_dir: Path, seed: int, rows_per_source: int) -> None:
    all_rows: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    for spec in DATASET_SPECS:
        source_rows = list(iter_rows(spec, seed=seed, limit=rows_per_source))
        by_source[spec.name] = len(source_rows)
        all_rows.extend(source_rows)

    rng = random.Random(seed)
    rng.shuffle(all_rows)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(all_rows)
    dataset.save_to_disk(str(output_dir / "dataset"))

    examples = render_examples(all_rows, max_examples=12)
    source_samples = all_rows[: min(40, len(all_rows))]

    save_json(
        output_dir / "metadata.json",
        {
            "mode": "demo",
            "seed": seed,
            "rows_per_source_requested": rows_per_source,
            "total_rows": len(all_rows),
            "rows_by_source": by_source,
            "source_specs": [spec.__dict__ for spec in DATASET_SPECS],
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                ]
            },
        },
    )
    save_jsonl(output_dir / "examples.jsonl", examples)
    save_jsonl(output_dir / "source_samples.jsonl", source_samples)


def _write_shuffled_dataset_jsonl(
    output_dir: Path,
    seed: int,
    row_iterables: list[Iterable[dict[str, Any]]],
    rows_by_source: dict[str, int],
    bytes_by_source: dict[str, int],
) -> Path:
    buckets_dir = output_dir / "_shuffle_buckets"
    buckets_dir.mkdir(parents=True, exist_ok=True)
    bucket_count = 64
    bucket_paths = [buckets_dir / f"bucket_{i:02d}.jsonl" for i in range(bucket_count)]
    bucket_files = [p.open("wb") for p in bucket_paths]

    try:
        for rows in row_iterables:
            for row in rows:
                source = row["source_dataset"]
                row_bytes = row_messages_bytes(row)
                rows_by_source[source] = rows_by_source.get(source, 0) + 1
                bytes_by_source[source] = bytes_by_source.get(source, 0) + row_bytes
                bucket_id = int(row["shuffle_key"][:8], 16) % bucket_count
                bucket_files[bucket_id].write(orjson.dumps(row))
                bucket_files[bucket_id].write(b"\n")
    finally:
        for f in bucket_files:
            f.close()

    shuffled_jsonl = output_dir / "dataset_shuffled.jsonl"
    bucket_order = list(range(bucket_count))
    random.Random(seed).shuffle(bucket_order)
    with shuffled_jsonl.open("wb") as out:
        for bucket_id in bucket_order:
            with bucket_paths[bucket_id].open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)

    shutil.rmtree(buckets_dir, ignore_errors=True)
    return shuffled_jsonl


def iter_rows_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            if isinstance(row, dict):
                yield row


def summarize_jsonl(path: Path) -> dict[str, Any]:
    rows_by_source: dict[str, int] = {}
    bytes_by_source: dict[str, int] = {}
    total_rows = 0
    for row in iter_rows_from_jsonl(path):
        source = str(row.get("source_dataset", "unknown"))
        rows_by_source[source] = rows_by_source.get(source, 0) + 1
        try:
            row_bytes = row_messages_bytes(row)
        except Exception:
            row_bytes = len(orjson.dumps(row.get("messages", [])))
        bytes_by_source[source] = bytes_by_source.get(source, 0) + row_bytes
        total_rows += 1
    return {
        "total_rows": total_rows,
        "rows_by_source": rows_by_source,
        "bytes_by_source": bytes_by_source,
    }


def build_additional(
    output_dir: Path,
    seed: int,
    additional_rows_per_split: int,
    materialize_hf_dataset: bool,
) -> None:
    if additional_rows_per_split <= 0:
        raise typer.BadParameter("additional_rows_per_split must be > 0")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_source: dict[str, int] = {}
    bytes_by_source: dict[str, int] = {}
    rows_by_split: dict[str, int] = {}
    bytes_by_split: dict[str, int] = {}

    def iter_spec_rows(spec: DatasetSpec) -> Iterable[dict[str, Any]]:
        spec_key = f"{spec.name}|{spec.config or 'default'}|{spec.split}"
        accepted = 0
        for row in iter_rows(spec, seed=seed, limit=additional_rows_per_split):
            row_bytes = row_messages_bytes(row)
            rows_by_split[spec_key] = rows_by_split.get(spec_key, 0) + 1
            bytes_by_split[spec_key] = bytes_by_split.get(spec_key, 0) + row_bytes
            accepted += 1
            yield row
        if accepted == 0:
            rows_by_split.setdefault(spec_key, 0)
            bytes_by_split.setdefault(spec_key, 0)

    shuffled_jsonl = _write_shuffled_dataset_jsonl(
        output_dir=output_dir,
        seed=seed,
        row_iterables=[iter_spec_rows(spec) for spec in ADDITIONAL_NEMOTRON_SPECS],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    if materialize_hf_dataset:
        ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
        ds.save_to_disk(str(output_dir / "dataset"))

    save_json(
        output_dir / "metadata.json",
        {
            "mode": "additional",
            "seed": seed,
            "additional_rows_per_split_requested": additional_rows_per_split,
            "total_rows": sum(rows_by_source.values()),
            "rows_by_source": rows_by_source,
            "bytes_by_source": bytes_by_source,
            "rows_by_split": rows_by_split,
            "bytes_by_split": bytes_by_split,
            "source_specs": [spec.__dict__ for spec in ADDITIONAL_NEMOTRON_SPECS],
            "materialize_hf_dataset": materialize_hf_dataset,
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                ]
            },
        },
    )


def build_extended_full(
    output_dir: Path,
    seed: int,
    base_full_jsonl: Path,
    additional_jsonl: Path,
    materialize_hf_dataset: bool,
) -> None:
    if not base_full_jsonl.exists():
        raise typer.BadParameter(f"base_full_input_jsonl not found: {base_full_jsonl}")
    if not additional_jsonl.exists():
        raise typer.BadParameter(f"additional_input_jsonl not found: {additional_jsonl}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_source: dict[str, int] = {}
    bytes_by_source: dict[str, int] = {}
    shuffled_jsonl = _write_shuffled_dataset_jsonl(
        output_dir=output_dir,
        seed=seed,
        row_iterables=[iter_rows_from_jsonl(base_full_jsonl), iter_rows_from_jsonl(additional_jsonl)],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    if materialize_hf_dataset:
        ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
        ds.save_to_disk(str(output_dir / "dataset"))

    base_summary = summarize_jsonl(base_full_jsonl)
    additional_summary = summarize_jsonl(additional_jsonl)
    save_json(
        output_dir / "metadata.json",
        {
            "mode": "extended_full",
            "seed": seed,
            "base_full_dataset_path": str(base_full_jsonl),
            "additional_dataset_path": str(additional_jsonl),
            "materialize_hf_dataset": materialize_hf_dataset,
            "total_rows": sum(rows_by_source.values()),
            "rows_by_source": rows_by_source,
            "bytes_by_source": bytes_by_source,
            "base_full_summary": base_summary,
            "additional_summary": additional_summary,
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                ]
            },
        },
    )


def build_full(output_dir: Path, seed: int) -> None:
    nvidia_source = "nvidia/Nemotron-Terminal-Corpus"
    nanbeige_source = "Nanbeige/ToolMind-Web-QA"
    swe_source = "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_source = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}
    bytes_by_source = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}

    shuffled_jsonl = _write_shuffled_dataset_jsonl(
        output_dir=output_dir,
        seed=seed,
        row_iterables=[
            (row for spec in NVIDIA_TERMINAL_SPECS for row in iter_rows(spec, seed=seed, limit=None)),
            iter_rows(DatasetSpec(nanbeige_source, "test", "train", "conversations"), seed=seed, limit=None),
            iter_rows(DatasetSpec(swe_source, "default", "train", "messages"), seed=seed, limit=None),
        ],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
    ds.save_to_disk(str(output_dir / "dataset"))

    total_rows = sum(rows_by_source.values())
    total_bytes = sum(bytes_by_source.values())
    nvidia_byte_share = (bytes_by_source[nvidia_source] / total_bytes) if total_bytes else 0.0

    balance_note = (
        "Included all NVIDIA terminal splits (easy/medium/mixed), all SWE rows, and all Nanbeige rows. "
        "Rows are deterministically shuffled via hash buckets."
    )

    save_json(
        output_dir / "metadata.json",
        {
            "mode": "full",
            "seed": seed,
            "total_rows": total_rows,
            "rows_by_source": rows_by_source,
            "rows_by_source_before_balance": rows_by_source,
            "bytes_by_source": bytes_by_source,
            "bytes_by_source_before_balance": bytes_by_source,
            "source_specs": [spec.__dict__ for spec in NVIDIA_TERMINAL_SPECS + DATASET_SPECS[1:]],
            "nvidia_minimum_byte_share": 0.5,
            "nvidia_actual_byte_share": nvidia_byte_share,
            "full_mode_balance_policy": "include_all_nvidia_all_swe_all_nanbeige",
            "full_mode_balance_note": balance_note,
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                ]
            },
        },
    )


def build_balanced(output_dir: Path, seed: int) -> None:
    nvidia_source = "nvidia/Nemotron-Terminal-Corpus"
    nanbeige_source = "Nanbeige/ToolMind-Web-QA"
    swe_source = "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    non_nvidia_rows: list[dict[str, Any]] = []
    rows_by_source_before_balance = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}
    bytes_by_source_before_balance = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}

    for spec in [
        DatasetSpec(nanbeige_source, "test", "train", "conversations"),
        DatasetSpec(swe_source, "default", "train", "messages"),
    ]:
        for row in iter_rows(spec, seed=seed, limit=None):
            non_nvidia_rows.append(row)
            src = row["source_dataset"]
            rows_by_source_before_balance[src] += 1
            bytes_by_source_before_balance[src] += row_messages_bytes(row)

    nvidia_rows_all: list[dict[str, Any]] = []
    for spec in NVIDIA_TERMINAL_SPECS:
        for row in iter_rows(spec, seed=seed, limit=None):
            nvidia_rows_all.append(row)
            rows_by_source_before_balance[nvidia_source] += 1
            bytes_by_source_before_balance[nvidia_source] += row_messages_bytes(row)

    non_nvidia_target_bytes = sum(row_messages_bytes(r) for r in non_nvidia_rows)
    nvidia_selected: list[dict[str, Any]] = []
    nvidia_bytes = 0
    best_extra_row: dict[str, Any] | None = None
    best_distance = abs(non_nvidia_target_bytes - nvidia_bytes)

    for row in sorted(nvidia_rows_all, key=lambda r: r["shuffle_key"]):
        row_bytes = row_messages_bytes(row)
        if nvidia_bytes + row_bytes <= non_nvidia_target_bytes:
            nvidia_selected.append(row)
            nvidia_bytes += row_bytes
            best_distance = abs(non_nvidia_target_bytes - nvidia_bytes)
        else:
            candidate_distance = abs(non_nvidia_target_bytes - (nvidia_bytes + row_bytes))
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_extra_row = row
    if best_extra_row is not None:
        nvidia_selected.append(best_extra_row)

    rows_by_source = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}
    bytes_by_source = {nvidia_source: 0, nanbeige_source: 0, swe_source: 0}
    shuffled_jsonl = _write_shuffled_dataset_jsonl(
        output_dir=output_dir,
        seed=seed,
        row_iterables=[iter(nvidia_selected), iter(non_nvidia_rows)],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
    ds.save_to_disk(str(output_dir / "dataset"))

    total_rows = sum(rows_by_source.values())
    total_bytes = sum(bytes_by_source.values())
    nvidia_byte_share = (bytes_by_source[nvidia_source] / total_bytes) if total_bytes else 0.0

    save_json(
        output_dir / "metadata.json",
        {
            "mode": "balanced",
            "seed": seed,
            "total_rows": total_rows,
            "rows_by_source": rows_by_source,
            "rows_by_source_before_balance": rows_by_source_before_balance,
            "bytes_by_source": bytes_by_source,
            "bytes_by_source_before_balance": bytes_by_source_before_balance,
            "source_specs": [spec.__dict__ for spec in NVIDIA_TERMINAL_SPECS + DATASET_SPECS[1:]],
            "nvidia_target_byte_share": 0.5,
            "nvidia_actual_byte_share": nvidia_byte_share,
            "full_mode_balance_policy": "all_swe_all_nanbeige_nvidia_to_50pct_bytes",
            "full_mode_balance_note": (
                "Included all SWE and Nanbeige rows, then selected NVIDIA rows to match non-NVIDIA bytes "
                "as closely as possible for ~50% NVIDIA by bytes."
            ),
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                ]
            },
        },
    )


@app.command("build")
def build(
    mode: str = typer.Option("demo", help="demo, full, balanced, additional, or extended_full"),
    seed: int = typer.Option(DEFAULT_SEED, help="Shuffle seed"),
    rows_per_source: int = typer.Option(100, help="Rows per source in demo mode"),
    demo_output_dir: Path = typer.Option(DEFAULT_DEMO_OUTPUT, help="Demo output directory"),
    full_output_dir: Path = typer.Option(DEFAULT_FULL_OUTPUT, help="Full output directory"),
    balanced_output_dir: Path = typer.Option(DEFAULT_BALANCED_OUTPUT, help="Balanced output directory"),
    additional_output_dir: Path = typer.Option(DEFAULT_ADDITIONAL_OUTPUT, help="Additional output directory"),
    extended_output_dir: Path = typer.Option(
        DEFAULT_EXTENDED_FULL_OUTPUT, help="Extended full output directory"
    ),
    additional_rows_per_split: int = typer.Option(10000, help="Rows per split in additional mode"),
    base_full_input_jsonl: Path = typer.Option(
        DEFAULT_FULL_OUTPUT / "dataset_shuffled.jsonl",
        help="Input JSONL from base full dataset for extended_full mode",
    ),
    additional_input_jsonl: Path = typer.Option(
        DEFAULT_ADDITIONAL_OUTPUT / "dataset_shuffled.jsonl",
        help="Input JSONL from additional dataset for extended_full mode",
    ),
    materialize_hf_dataset: bool = typer.Option(
        False,
        help="If true, also save a Hugging Face dataset/ directory for additional and extended_full modes",
    ),
) -> None:
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"demo", "full", "balanced", "additional", "extended_full"}:
        raise typer.BadParameter("mode must be one of: demo, full, balanced, additional, extended_full")

    if normalized_mode == "demo":
        build_demo(output_dir=demo_output_dir, seed=seed, rows_per_source=rows_per_source)
        typer.echo(f"Demo dataset written to {demo_output_dir}")
        return

    if normalized_mode == "balanced":
        build_balanced(output_dir=balanced_output_dir, seed=seed)
        typer.echo(f"Balanced dataset written to {balanced_output_dir}")
        return

    if normalized_mode == "additional":
        build_additional(
            output_dir=additional_output_dir,
            seed=seed,
            additional_rows_per_split=additional_rows_per_split,
            materialize_hf_dataset=materialize_hf_dataset,
        )
        typer.echo(f"Additional dataset written to {additional_output_dir}")
        return

    if normalized_mode == "extended_full":
        build_extended_full(
            output_dir=extended_output_dir,
            seed=seed,
            base_full_jsonl=base_full_input_jsonl,
            additional_jsonl=additional_input_jsonl,
            materialize_hf_dataset=materialize_hf_dataset,
        )
        typer.echo(f"Extended full dataset written to {extended_output_dir}")
        return

    build_full(output_dir=full_output_dir, seed=seed)
    typer.echo(f"Full dataset written to {full_output_dir}")


if __name__ == "__main__":
    app()
