from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import requests
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import orjson
import typer
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

app = typer.Typer(
    add_completion=False, help="Build Qwen SFT datasets in messages format."
)

DEFAULT_DEMO_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/sft/data/demo-dataset")
DEFAULT_FULL_OUTPUT = Path("/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset")
DEFAULT_BALANCED_OUTPUT = Path(
    "/wbl-fast/usrs/ethan/small-agent/sft/data/balanced-dataset"
)
DEFAULT_ADDITIONAL_OUTPUT = Path(
    "/wbl-fast/usrs/ethan/small-agent/sft/data/additional-data"
)
DEFAULT_EXTENDED_FULL_OUTPUT = Path(
    "/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset-extended"
)
DEFAULT_SEED = 42
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_ESCAPED_BLOCK_RE = re.compile(
    r"&lt;think&gt;.*?&lt;/think&gt;", flags=re.DOTALL | re.IGNORECASE
)
THINK_OPEN_RE = re.compile(r"<think>|&lt;think&gt;", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>|&lt;/think&gt;", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: str | None
    split: str
    messages_key: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "nvidia/Nemotron-Terminal-Corpus", "skill_based_mixed", "train", "conversations"
    ),
    DatasetSpec("Nanbeige/ToolMind-Web-QA", "test", "train", "conversations"),
    DatasetSpec(
        "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K",
        "default",
        "train",
        "messages",
    ),
)

NVIDIA_TERMINAL_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "nvidia/Nemotron-Terminal-Corpus", "skill_based_easy", "train", "conversations"
    ),
    DatasetSpec(
        "nvidia/Nemotron-Terminal-Corpus",
        "skill_based_medium",
        "train",
        "conversations",
    ),
    DatasetSpec(
        "nvidia/Nemotron-Terminal-Corpus", "skill_based_mixed", "train", "conversations"
    ),
)

ADDITIONAL_NEMOTRON_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "nvidia/Nemotron-Agentic-v1", "default", "interactive_agent", "messages"
    ),
    DatasetSpec("nvidia/Nemotron-Agentic-v1", "default", "tool_calling", "messages"),
    DatasetSpec(
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "default",
        "chat_if",
        "messages",
    ),
    DatasetSpec(
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "default",
        "structured_outputs",
        "messages",
    ),
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
        return load_dataset(
            "json", data_files=[data_file], split="train", streaming=True
        )
    if spec.name == "nvidia/Nemotron-Instruction-Following-Chat-v1":
        split_to_file = {
            "chat_if": "hf://datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/data/chat_if.jsonl",
            "structured_outputs": "hf://datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/data/structured_outputs.jsonl",
        }
        data_file = split_to_file.get(spec.split)
        if data_file is None:
            raise ValueError(f"Unsupported split for {spec.name}: {spec.split}")
        return load_dataset(
            "json", data_files=[data_file], split="train", streaming=True
        )
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


def use_agent_canonicalization(spec: DatasetSpec) -> bool:
    return spec.name in {
        "nvidia/Nemotron-Agentic-v1",
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
    }


def _canonical_agent_role(role: str) -> Optional[str]:
    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool_call": "tool_call",
        "function_call": "tool_call",
        "tool_response": "tool_response",
        "tool": "tool_response",
        "function_response": "tool_response",
        "observation": "tool_response",
        "observations": "tool_response",
    }
    key = role.strip().lower().replace("-", "_")
    return role_map.get(key)


def _canonicalize_tools_field(tools: Any) -> tuple[Optional[str], Optional[str]]:
    if tools is None:
        return None, None
    parsed: Any
    if isinstance(tools, str):
        try:
            parsed = orjson.loads(tools)
        except Exception:
            return None, "invalid_tools_json"
    elif isinstance(tools, (list, tuple)):
        parsed = list(tools)
    elif isinstance(tools, dict):
        parsed = [tools]
    else:
        return None, "invalid_tools_type"

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None, "invalid_tools_payload"
    try:
        return json.dumps(parsed, ensure_ascii=False), None
    except Exception:
        return None, "invalid_tools_payload"


def _canonicalize_tool_content(
    content: Any, role: str
) -> tuple[Optional[str], Optional[str]]:
    payload: Any = content
    if isinstance(content, str):
        try:
            payload = orjson.loads(content)
        except Exception:
            if role == "tool_response":
                payload = {"raw_output": content}
            else:
                return None, "invalid_tool_call_json"
    elif isinstance(content, (dict, list)):
        payload = content
    else:
        if role == "tool_response":
            payload = {"raw_output": str(content)}
        else:
            return None, "invalid_tool_call_payload"

    if role == "tool_call":
        if not isinstance(payload, dict):
            return None, "invalid_tool_call_payload"
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return None, "missing_tool_name"
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = orjson.loads(arguments)
            except Exception:
                return None, "invalid_tool_arguments"
        if not isinstance(arguments, dict):
            return None, "invalid_tool_arguments"
        payload = {"name": tool_name, "arguments": arguments}

    try:
        return json.dumps(payload, ensure_ascii=False), None
    except Exception:
        return None, "invalid_tool_content"


def canonicalize_agent_row(
    messages: list[dict[str, str]], tools: Any
) -> tuple[Optional[list[dict[str, str]]], Optional[str], Optional[str]]:
    has_tool_messages = False
    canonical_messages: list[dict[str, str]] = []
    pending_tool_call = False

    for idx, message in enumerate(messages):
        role_raw = str(message.get("role", ""))
        role = _canonical_agent_role(role_raw)
        if role is None:
            return None, None, f"unsupported_role:{role_raw}"

        content: Any = message.get("content", "")
        if role in {"tool_call", "tool_response"}:
            has_tool_messages = True
            content, content_err = _canonicalize_tool_content(content, role)
            if content_err:
                return None, None, content_err
        else:
            content = to_text(content).strip()

        if not content:
            continue

        if role == "system" and idx != 0:
            return None, None, "system_role_not_first"
        if role == "tool_call":
            pending_tool_call = True
        elif role == "tool_response":
            if not pending_tool_call:
                return None, None, "missing_tool_call_before_tool_response"
        elif role == "user":
            pending_tool_call = False
        elif role == "assistant":
            pending_tool_call = False

        canonical_messages.append({"role": role, "content": content})

    normalized_tools: Any = tools
    tools_err: Optional[str] = None
    if has_tool_messages:
        normalized_tools, tools_err = _canonicalize_tools_field(tools)
    if has_tool_messages and normalized_tools is None:
        reason = tools_err or "missing_tools_for_tool_messages"
        return None, None, reason
    if not canonical_messages:
        return None, None, "empty_messages_after_canonicalization"
    return canonical_messages, normalized_tools, None


def _summarize_agent_row_changes(
    before_messages: list[dict[str, str]],
    after_messages: list[dict[str, str]],
    before_tools: Any,
    after_tools: Any,
) -> Optional[dict[str, Any]]:
    before_roles = [
        str(m.get("role", "")) for m in before_messages if isinstance(m, dict)
    ]
    after_roles = [
        str(m.get("role", "")) for m in after_messages if isinstance(m, dict)
    ]
    role_changed = before_roles != after_roles
    message_count_changed = len(before_messages) != len(after_messages)
    tool_field_changed = before_tools != after_tools

    if not role_changed and not message_count_changed and not tool_field_changed:
        return None
    return {
        "before_roles": before_roles,
        "after_roles": after_roles,
        "before_message_count": len(before_messages),
        "after_message_count": len(after_messages),
        "tool_field_changed": tool_field_changed,
    }


def validate_row_for_megatron_agent_training(row: dict[str, Any]) -> Optional[str]:
    messages = deepcopy(row.get("messages", []))
    if not isinstance(messages, list) or not messages:
        return "invalid_messages"

    for message in messages:
        role = message.get("role")
        if role in {"tool_call", "tool_response", "tool"}:
            content = message.get("content")
            if not isinstance(content, str):
                return "non_string_tool_content"
            try:
                orjson.loads(content)
            except Exception:
                return "non_json_tool_content"

    if messages and messages[0].get("role") == "system":
        messages = messages[1:]

    compact: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role == "tool_call":
            j = i
            while j + 1 < len(messages) and messages[j + 1].get("role") == "tool_call":
                j += 1
            compact.append({"role": "assistant", "content": "<tool_call_block>"})
            i = j + 1
            continue
        if role == "tool_response":
            role = "tool"
        compact.append({"role": role, "content": msg.get("content")})
        i += 1

    i = 1
    while i < len(compact):
        pre_role = compact[i - 1].get("role")
        role = compact[i].get("role")
        if pre_role == "assistant" and role == "tool":
            j = i
            while j + 1 < len(compact) and compact[j + 1].get("role") == "tool":
                j += 1
            compact[i : j + 1] = [{"role": "tool", "content": "<tool_response_block>"}]
            i += 1
            continue
        if (pre_role, role) in {("assistant", "assistant"), ("user", "user")}:
            compact[i - 1]["content"] = (
                f"{compact[i - 1].get('content', '')}{compact[i].get('content', '')}"
            )
            compact.pop(i)
            continue
        i += 1

    if compact and compact[0].get("role") == "assistant":
        compact.insert(0, {"role": "user", "content": ""})
    if len(compact) % 2 == 1:
        compact.append({"role": "assistant", "content": None})

    for query_message, response_message in zip(compact[::2], compact[1::2]):
        query_role = query_message.get("role")
        response_role = response_message.get("role")
        if query_role not in {"user", "tool"}:
            return f"invalid_query_role:{query_role}"
        if response_role != "assistant":
            return f"invalid_response_role:{response_role}"
    return None


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


def iter_rows_from_jsonl_url(
    spec: DatasetSpec,
    seed: int,
    limit: int | None = None,
    enforce_agent_canonical: bool = False,
    on_row_dropped: Optional[Callable[[DatasetSpec, str, str], None]] = None,
    on_row_changed: Optional[Callable[[DatasetSpec, str, dict[str, Any]], None]] = None,
) -> Iterable[dict[str, Any]]:
    url = additional_spec_jsonl_url(spec)
    if url is None:
        raise ValueError(
            f"No JSONL URL mapping for dataset spec: {spec.name}/{spec.split}"
        )

    normalize = (
        normalize_messages_lightweight
        if use_lightweight_normalization(spec)
        else normalize_messages
    )
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
            tools = row.get("tools")
            source_config = spec.config or "default"
            generated_row_id = f"{spec.name}:{source_config}:{spec.split}:{row_id}"
            if enforce_agent_canonical and use_agent_canonicalization(spec):
                before_messages = deepcopy(messages)
                before_tools = tools
                messages, tools, drop_reason = canonicalize_agent_row(messages, tools)
                if drop_reason is not None:
                    if on_row_dropped is not None:
                        on_row_dropped(spec, generated_row_id, drop_reason)
                    continue
                assert messages is not None
                if on_row_changed is not None:
                    change_summary = _summarize_agent_row_changes(
                        before_messages, messages, before_tools, tools
                    )
                    if change_summary is not None:
                        on_row_changed(spec, generated_row_id, change_summary)
            row_payload: dict[str, Any] = {
                "source_dataset": spec.name,
                "source_config": source_config,
                "source_split": spec.split,
                "row_id": generated_row_id,
                "shuffle_key": _shuffle_key(
                    seed, spec.name, source_config, spec.split, row_id
                ),
                "messages": messages,
            }
            if tools is not None:
                row_payload["tools"] = tools
            if enforce_agent_canonical and use_agent_canonicalization(spec):
                validation_error = validate_row_for_megatron_agent_training(row_payload)
                if validation_error is not None:
                    if on_row_dropped is not None:
                        on_row_dropped(spec, generated_row_id, validation_error)
                    continue
            yield row_payload
            accepted += 1
            if limit is not None and accepted >= limit:
                break


def _shuffle_key(
    seed: int, source_dataset: str, source_config: str, source_split: str, row_id: int
) -> str:
    payload = f"{seed}:{source_dataset}:{source_config}:{source_split}:{row_id}".encode(
        "utf-8"
    )
    return hashlib.sha1(payload).hexdigest()


def iter_rows(
    spec: DatasetSpec,
    seed: int,
    limit: int | None = None,
    enforce_agent_canonical: bool = False,
    on_row_dropped: Optional[Callable[[DatasetSpec, str, str], None]] = None,
    on_row_changed: Optional[Callable[[DatasetSpec, str, dict[str, Any]], None]] = None,
) -> Iterable[dict[str, Any]]:
    direct_jsonl_url = additional_spec_jsonl_url(spec)
    if direct_jsonl_url is not None:
        yield from iter_rows_from_jsonl_url(
            spec,
            seed=seed,
            limit=limit,
            enforce_agent_canonical=enforce_agent_canonical,
            on_row_dropped=on_row_dropped,
            on_row_changed=on_row_changed,
        )
        return

    ds = get_streaming_dataset(spec)
    normalize = (
        normalize_messages_lightweight
        if use_lightweight_normalization(spec)
        else normalize_messages
    )
    accepted = 0
    for row_id, row in enumerate(ds):
        messages = normalize(row.get(spec.messages_key))
        if not messages:
            continue
        tools = row.get("tools")
        source_config = spec.config or "default"
        generated_row_id = f"{spec.name}:{source_config}:{spec.split}:{row_id}"
        if enforce_agent_canonical and use_agent_canonicalization(spec):
            before_messages = deepcopy(messages)
            before_tools = tools
            messages, tools, drop_reason = canonicalize_agent_row(messages, tools)
            if drop_reason is not None:
                if on_row_dropped is not None:
                    on_row_dropped(spec, generated_row_id, drop_reason)
                continue
            assert messages is not None
            if on_row_changed is not None:
                change_summary = _summarize_agent_row_changes(
                    before_messages, messages, before_tools, tools
                )
                if change_summary is not None:
                    on_row_changed(spec, generated_row_id, change_summary)
        row_payload: dict[str, Any] = {
            "source_dataset": spec.name,
            "source_config": source_config,
            "source_split": spec.split,
            "row_id": generated_row_id,
            "shuffle_key": _shuffle_key(
                seed, spec.name, source_config, spec.split, row_id
            ),
            "messages": messages,
        }
        if tools is not None:
            row_payload["tools"] = tools
        if enforce_agent_canonical and use_agent_canonicalization(spec):
            validation_error = validate_row_for_megatron_agent_training(row_payload)
            if validation_error is not None:
                if on_row_dropped is not None:
                    on_row_dropped(spec, generated_row_id, validation_error)
                continue
        yield row_payload
        accepted += 1
        if limit is not None and accepted >= limit:
            break


def render_examples(
    rows: list[dict[str, Any]], max_examples: int = 12
) -> list[dict[str, Any]]:
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("ab") as f:
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
                # Keep a stable JSON schema for HF loader: always emit `tools` as JSON string.
                tools_value = row.get("tools", None)
                if tools_value is None:
                    row["tools"] = "[]"
                elif not isinstance(tools_value, str):
                    row["tools"] = json.dumps(tools_value, ensure_ascii=False)
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


def _load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_agent_jsonl(path: Path) -> dict[str, Any]:
    checked_rows = 0
    agent_rows = 0
    for row in iter_rows_from_jsonl(path):
        checked_rows += 1
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Invalid messages field in {path}: row={checked_rows}")
        if not any(
            m.get("role") in {"tool", "tool_response", "tool_call"}
            for m in messages
            if isinstance(m, dict)
        ):
            continue
        agent_rows += 1
        validation_error = validate_row_for_megatron_agent_training(row)
        if validation_error is not None:
            row_id = row.get("row_id", f"line_{checked_rows}")
            raise ValueError(
                f"Agent preflight validation failed for {path} row_id={row_id}: {validation_error}"
            )
    return {"checked_rows": checked_rows, "agent_rows": agent_rows}


def _row_signature(row: dict[str, Any]) -> str:
    payload = {
        "messages": row.get("messages"),
        "tools": row.get("tools"),
    }
    return hashlib.sha1(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def diff_jsonl_by_row_id(old_path: Optional[Path], new_path: Path) -> dict[str, Any]:
    old_rows: dict[str, str] = {}
    if old_path is not None and old_path.exists():
        for row in iter_rows_from_jsonl(old_path):
            row_id = str(row.get("row_id", ""))
            if row_id:
                old_rows[row_id] = _row_signature(row)

    new_rows: dict[str, str] = {}
    for row in iter_rows_from_jsonl(new_path):
        row_id = str(row.get("row_id", ""))
        if row_id:
            new_rows[row_id] = _row_signature(row)

    removed = sorted([row_id for row_id in old_rows.keys() if row_id not in new_rows])
    added = sorted([row_id for row_id in new_rows.keys() if row_id not in old_rows])
    changed = sorted(
        [
            row_id
            for row_id in old_rows.keys() & new_rows.keys()
            if old_rows[row_id] != new_rows[row_id]
        ]
    )
    return {
        "old_row_count": len(old_rows),
        "new_row_count": len(new_rows),
        "removed_row_ids": removed,
        "added_row_ids": added,
        "changed_row_ids": changed,
    }


def build_additional(
    output_dir: Path,
    seed: int,
    additional_rows_per_split: int,
    materialize_hf_dataset: bool,
) -> None:
    if additional_rows_per_split <= 0:
        raise typer.BadParameter("additional_rows_per_split must be > 0")

    previous_jsonl = output_dir / "dataset_shuffled.jsonl"
    previous_jsonl_backup: Optional[Path] = None
    if previous_jsonl.exists():
        previous_jsonl_backup = (
            output_dir.parent / f".{output_dir.name}.previous.dataset_shuffled.jsonl"
        )
        shutil.copy2(previous_jsonl, previous_jsonl_backup)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    changed_rows_log = output_dir / "rows_changed_canonicalization.jsonl"
    removed_rows_log = output_dir / "rows_removed_canonicalization.jsonl"

    rows_by_source: dict[str, int] = {}
    bytes_by_source: dict[str, int] = {}
    rows_by_split: dict[str, int] = {}
    bytes_by_split: dict[str, int] = {}
    dropped_rows_by_split: dict[str, int] = {}
    dropped_rows_by_split_and_reason: dict[str, dict[str, int]] = {}

    def on_row_dropped(spec: DatasetSpec, row_id: str, reason: str) -> None:
        spec_key = f"{spec.name}|{spec.config or 'default'}|{spec.split}"
        dropped_rows_by_split[spec_key] = dropped_rows_by_split.get(spec_key, 0) + 1
        by_reason = dropped_rows_by_split_and_reason.setdefault(spec_key, {})
        by_reason[reason] = by_reason.get(reason, 0) + 1
        append_jsonl(
            removed_rows_log,
            {
                "row_id": row_id,
                "source_dataset": spec.name,
                "source_config": spec.config or "default",
                "source_split": spec.split,
                "reason": reason,
            },
        )

    def on_row_changed(
        spec: DatasetSpec, row_id: str, change_summary: dict[str, Any]
    ) -> None:
        append_jsonl(
            changed_rows_log,
            {
                "row_id": row_id,
                "source_dataset": spec.name,
                "source_config": spec.config or "default",
                "source_split": spec.split,
                **change_summary,
            },
        )

    def iter_spec_rows(spec: DatasetSpec) -> Iterable[dict[str, Any]]:
        spec_key = f"{spec.name}|{spec.config or 'default'}|{spec.split}"
        accepted = 0
        for row in iter_rows(
            spec,
            seed=seed,
            limit=additional_rows_per_split,
            enforce_agent_canonical=True,
            on_row_dropped=on_row_dropped,
            on_row_changed=on_row_changed,
        ):
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
        ds.save_to_disk(str(output_dir / "dataset"))  # type: ignore

    preflight_summary = validate_agent_jsonl(shuffled_jsonl)
    rebuild_diff = diff_jsonl_by_row_id(
        old_path=previous_jsonl_backup
        if previous_jsonl_backup and previous_jsonl_backup.exists()
        else None,
        new_path=shuffled_jsonl,
    )
    if previous_jsonl_backup is not None and previous_jsonl_backup.exists():
        previous_jsonl_backup.unlink()
    save_json(
        output_dir / "rebuild_row_diff.json",
        rebuild_diff,
    )
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
            "rows_dropped_invalid_agent_format": sum(dropped_rows_by_split.values()),
            "rows_dropped_by_split": dropped_rows_by_split,
            "rows_dropped_by_split_and_reason": dropped_rows_by_split_and_reason,
            "row_audit_logs": {
                "rows_changed_canonicalization": str(changed_rows_log),
                "rows_removed_canonicalization": str(removed_rows_log),
                "rebuild_row_diff": str(output_dir / "rebuild_row_diff.json"),
            },
            "agent_preflight_validation": preflight_summary,
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
                    "tools (optional)",
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
        raise typer.BadParameter(
            f"additional_input_jsonl not found: {additional_jsonl}"
        )

    previous_jsonl = output_dir / "dataset_shuffled.jsonl"
    previous_jsonl_backup: Optional[Path] = None
    if previous_jsonl.exists():
        previous_jsonl_backup = (
            output_dir.parent / f".{output_dir.name}.previous.dataset_shuffled.jsonl"
        )
        shutil.copy2(previous_jsonl, previous_jsonl_backup)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_source: dict[str, int] = {}
    bytes_by_source: dict[str, int] = {}
    shuffled_jsonl = _write_shuffled_dataset_jsonl(
        output_dir=output_dir,
        seed=seed,
        row_iterables=[
            iter_rows_from_jsonl(base_full_jsonl),
            iter_rows_from_jsonl(additional_jsonl),
        ],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    if materialize_hf_dataset:
        ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
        ds.save_to_disk(str(output_dir / "dataset"))  # type: ignore

    preflight_summary = validate_agent_jsonl(shuffled_jsonl)
    rebuild_diff = diff_jsonl_by_row_id(
        old_path=previous_jsonl_backup
        if previous_jsonl_backup and previous_jsonl_backup.exists()
        else None,
        new_path=shuffled_jsonl,
    )
    if previous_jsonl_backup is not None and previous_jsonl_backup.exists():
        previous_jsonl_backup.unlink()
    save_json(output_dir / "rebuild_row_diff.json", rebuild_diff)

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
            "agent_preflight_validation": preflight_summary,
            "row_audit_logs": {
                "rebuild_row_diff": str(output_dir / "rebuild_row_diff.json"),
            },
            "schema": {
                "fields": [
                    "source_dataset",
                    "source_config",
                    "source_split",
                    "row_id",
                    "shuffle_key",
                    "messages",
                    "tools (optional)",
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
            (
                row
                for spec in NVIDIA_TERMINAL_SPECS
                for row in iter_rows(spec, seed=seed, limit=None)
            ),
            iter_rows(
                DatasetSpec(nanbeige_source, "test", "train", "conversations"),
                seed=seed,
                limit=None,
            ),
            iter_rows(
                DatasetSpec(swe_source, "default", "train", "messages"),
                seed=seed,
                limit=None,
            ),
        ],
        rows_by_source=rows_by_source,
        bytes_by_source=bytes_by_source,
    )

    ds = load_dataset("json", data_files=str(shuffled_jsonl), split="train")
    ds.save_to_disk(str(output_dir / "dataset"))  # type: ignore

    total_rows = sum(rows_by_source.values())
    total_bytes = sum(bytes_by_source.values())
    nvidia_byte_share = (
        (bytes_by_source[nvidia_source] / total_bytes) if total_bytes else 0.0
    )

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
            "source_specs": [
                spec.__dict__ for spec in NVIDIA_TERMINAL_SPECS + DATASET_SPECS[1:]
            ],
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
    rows_by_source_before_balance = {
        nvidia_source: 0,
        nanbeige_source: 0,
        swe_source: 0,
    }
    bytes_by_source_before_balance = {
        nvidia_source: 0,
        nanbeige_source: 0,
        swe_source: 0,
    }

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
            candidate_distance = abs(
                non_nvidia_target_bytes - (nvidia_bytes + row_bytes)
            )
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
    ds.save_to_disk(str(output_dir / "dataset"))  # type: ignore

    total_rows = sum(rows_by_source.values())
    total_bytes = sum(bytes_by_source.values())
    nvidia_byte_share = (
        (bytes_by_source[nvidia_source] / total_bytes) if total_bytes else 0.0
    )

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
            "source_specs": [
                spec.__dict__ for spec in NVIDIA_TERMINAL_SPECS + DATASET_SPECS[1:]
            ],
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
    mode: str = typer.Option(
        "demo", help="demo, full, balanced, additional, or extended_full"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="Shuffle seed"),
    rows_per_source: int = typer.Option(100, help="Rows per source in demo mode"),
    demo_output_dir: Path = typer.Option(
        DEFAULT_DEMO_OUTPUT, help="Demo output directory"
    ),
    full_output_dir: Path = typer.Option(
        DEFAULT_FULL_OUTPUT, help="Full output directory"
    ),
    balanced_output_dir: Path = typer.Option(
        DEFAULT_BALANCED_OUTPUT, help="Balanced output directory"
    ),
    additional_output_dir: Path = typer.Option(
        DEFAULT_ADDITIONAL_OUTPUT, help="Additional output directory"
    ),
    extended_output_dir: Path = typer.Option(
        DEFAULT_EXTENDED_FULL_OUTPUT, help="Extended full output directory"
    ),
    additional_rows_per_split: int = typer.Option(
        10000, help="Rows per split in additional mode"
    ),
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
    if normalized_mode not in {
        "demo",
        "full",
        "balanced",
        "additional",
        "extended_full",
    }:
        raise typer.BadParameter(
            "mode must be one of: demo, full, balanced, additional, extended_full"
        )

    if normalized_mode == "demo":
        build_demo(
            output_dir=demo_output_dir, seed=seed, rows_per_source=rows_per_source
        )
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
