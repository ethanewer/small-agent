from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.core.events import AgentEvent
from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig
from agents.local_binary import resolve_agent_binary
from agents.openai_compat import (
    normalize_openai_compatible_model,
    preflight_agent_model_compatibility,
)
from agents.qwen.util import run_subprocess


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default

    return int(value)


def _qwen_actionable_error_message(
    *,
    model: str,
    process_error: subprocess.CalledProcessError,
) -> str | None:
    stderr = str(process_error.stderr or "")
    stdout = str(process_error.output or "")
    combined = f"{stdout}\n{stderr}".lower()
    is_chat_mismatch = (
        "not a chat model" in combined
        or "use v1/completions" in combined
        or "chat.completions" in combined
        and "not supported" in combined
    )
    if not is_chat_mismatch:
        return None

    return (
        f"Qwen Code cannot use model '{model}' via Chat Completions on this endpoint.\n"
        "Pick a chat-completions-compatible model (for example: gpt-4.1, gpt-4o-mini, "
        "or an OpenRouter qwen/* chat route), or run this model with the "
        "`terminus-2` agent if you need completion-style workflows."
    )


def _as_json(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _format_tool_input(arguments: Any) -> str:
    if arguments is None:
        return "{}"

    if isinstance(arguments, str):
        return arguments

    try:
        compact = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
    except Exception:  # noqa: BLE001
        return _as_json(arguments)

    if len(compact) <= 100:
        return compact

    return json.dumps(arguments, ensure_ascii=True, sort_keys=True, indent=2)


def _compact_single_line(text: str, *, max_chars: int = 160) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line

    return single_line[: max_chars - 3] + "..."


def _extract_tool_name(payload: dict[str, Any]) -> str:
    direct_name = payload.get("name") or payload.get("tool_name")
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name

    tool_obj = payload.get("tool")
    if isinstance(tool_obj, dict):
        tool_name = tool_obj.get("name")
        if isinstance(tool_name, str) and tool_name.strip():
            return tool_name

        function_obj = tool_obj.get("function")
        if isinstance(function_obj, dict):
            function_name = function_obj.get("name")
            if isinstance(function_name, str) and function_name.strip():
                return function_name

    function_obj = payload.get("function")
    if isinstance(function_obj, dict):
        function_name = function_obj.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name

    payload_id = payload.get("id")
    if isinstance(payload_id, str) and payload_id.strip():
        return payload_id

    return "tool"


def _extract_tool_arguments(payload: dict[str, Any]) -> Any:
    direct = (
        payload.get("arguments")
        or payload.get("input")
        or payload.get("args")
        or payload.get("parameters")
    )
    if direct is not None:
        return direct

    tool_obj = payload.get("tool")
    if isinstance(tool_obj, dict):
        nested = (
            tool_obj.get("arguments")
            or tool_obj.get("input")
            or tool_obj.get("args")
            or tool_obj.get("parameters")
        )
        if nested is not None:
            return nested

        function_obj = tool_obj.get("function")
        if isinstance(function_obj, dict):
            fn_args = (
                function_obj.get("arguments")
                or function_obj.get("input")
                or function_obj.get("parameters")
            )
            if fn_args is not None:
                return fn_args

    function_obj = payload.get("function")
    if isinstance(function_obj, dict):
        fn_args = (
            function_obj.get("arguments")
            or function_obj.get("input")
            or function_obj.get("parameters")
        )
        if fn_args is not None:
            return fn_args

    return None


def _iter_message_content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []

    content = message.get("content")
    if not isinstance(content, list):
        return []

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            blocks.append(item)
    return blocks


def _emit_qwen_stream_event(
    *,
    event: dict[str, Any],
    console: Console,
    sink: EventSink | None,
    verbosity: int,
    pending_tool_calls: list[dict[str, Any]],
) -> str | None:
    event_type = str(event.get("type", "")).strip().lower()
    subtype = str(event.get("subtype", "")).strip().lower()

    text_blocks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in _iter_message_content_blocks(event=event):
        block_type = str(block.get("type", "")).strip().lower()
        if block_type in {"text", "output_text"}:
            text = str(block.get("text", "")).strip()
            if text:
                text_blocks.append(text)

            continue

        if "tool" in block_type and ("call" in block_type or "use" in block_type):
            tool_calls.append(block)
            continue

        if "tool" in block_type and "result" in block_type:
            tool_results.append(block)

    if event_type in {"tool_call", "tool_use"}:
        tool_calls.append(event)
    elif event_type in {"tool_result", "tool_output"}:
        tool_results.append(event)

    if text_blocks:
        full_text = "\n".join(text_blocks)
        if sink:
            sink.emit(
                event=AgentEvent(
                    event_type="reasoning",
                    payload={
                        "message": full_text,
                        "source": "qwen",
                    },
                )
            )

        return full_text

    for call in tool_calls:
        name = _extract_tool_name(payload=call)
        call_id = str(
            call.get("id") or call.get("tool_call_id") or call.get("call_id") or ""
        )
        arguments = _extract_tool_arguments(payload=call)
        pending_tool_calls.append(
            {
                "name": name,
                "id": call_id,
                "arguments": arguments,
            }
        )
        if sink:
            sink.emit(
                event=AgentEvent(
                    event_type="tool_call",
                    payload={"name": name, "arguments": arguments, "raw": call},
                )
            )

    for result in tool_results:
        result_name = _extract_tool_name(payload=result)
        result_call_id = str(
            result.get("tool_call_id")
            or result.get("call_id")
            or result.get("id")
            or ""
        )
        matched_idx = None
        if result_call_id:
            for idx, pending in enumerate(pending_tool_calls):
                pending_id = str(pending.get("id") or "")
                if pending_id and pending_id == result_call_id:
                    matched_idx = idx
                    break

        if matched_idx is None:
            for idx, pending in enumerate(pending_tool_calls):
                if str(pending.get("name", "")) == result_name:
                    matched_idx = idx
                    break

        if matched_idx is None and pending_tool_calls:
            # Some providers emit generic tool results without stable ids/names.
            # In that case, pair in FIFO order to keep call/output blocks grouped.
            matched_idx = 0

        if matched_idx is not None:
            pending_call = pending_tool_calls.pop(matched_idx)
            display_name = str(pending_call.get("name") or result_name)
            display_input = _format_tool_input(arguments=pending_call.get("arguments"))
        else:
            display_name = result_name
            display_input = "{}"

        output = (
            result.get("output")
            or result.get("result")
            or result.get("content")
            or result.get("text")
        )
        output_text = _as_json(output).strip()
        has_meaningful_output = bool(output_text) and output_text.lower() != "tool"
        if verbosity == 0:
            console.print(f"{display_name}: {_compact_single_line(display_input)}")
            if has_meaningful_output:
                console.print(_compact_single_line(output_text))

            console.print("─" * max(20, console.width), style="dim")
        else:
            console.print(f"{display_name}: {display_input}")
            if has_meaningful_output:
                console.print(output_text)
            else:
                console.print("[no detailed output]")
            console.print("─" * max(20, console.width), style="dim")
        if sink:
            sink.emit(
                event=AgentEvent(
                    event_type="tool_result",
                    payload={"name": result_name, "output": output, "raw": result},
                )
            )

    if event_type == "assistant" and subtype in {"thinking", "reasoning"}:
        thinking = str(event.get("text") or event.get("message") or "").strip()
        if thinking:
            if verbosity == 0:
                console.print(f"assistant: {thinking}")
            else:
                console.print(
                    Panel(thinking, title="Assistant", border_style="magenta")
                )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="reasoning",
                        payload={"message": thinking, "source": "qwen"},
                    )
                )

            return thinking

    return None


class QwenHeadlessAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        result = self.run_task(
            task=Task.from_instruction(instruction=instruction),
            cfg=cfg,
            console=console,
            sink=None,
        )
        return result.exit_code

    def run_task(
        self,
        *,
        task: Task,
        cfg: AgentRuntimeConfig,
        console: Console | None = None,
        sink: EventSink | None = None,
    ) -> RunResult:
        if console is None:
            console = Console()

        options = cfg.agent_config
        verbosity = int(options.get("verbosity", 1))
        binary = str(
            options.get("binary") or resolve_agent_binary(default_binary="qwen")
        )
        token_limit = _coerce_int(value=options.get("token_limit"), default=131072)
        sampling_params = dict(options.get("sampling_params", {}))
        mcp_servers = dict(options.get("mcp_servers", {}))
        normalized_model = normalize_openai_compatible_model(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        compatibility_error = preflight_agent_model_compatibility(
            agent_key="qwen",
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        if compatibility_error:
            console.print(
                Panel(
                    compatibility_error,
                    title="Agent Compatibility Error",
                    border_style="red",
                )
            )
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message=compatibility_error,
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={
                            "kind": "compatibility",
                            "message": compatibility_error,
                        },
                    )
                )
                sink.finalize(result=result)

            return result

        settings: dict[str, Any] = {
            "selectedAuthType": "openai",
            "sessionTokenLimit": token_limit,
        }
        if sampling_params:
            settings["sampling_params"] = sampling_params

        if mcp_servers:
            settings["mcpServers"] = mcp_servers

        env = {
            "OPENAI_MODEL": normalized_model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_API_BASE": cfg.model.api_base,
            "OPENAI_API_KEY": cfg.model.api_key,
            "PATH": os.environ.get("PATH", ""),
            "NODE_NO_WARNINGS": "1",
            **{key: str(val) for key, val in dict(options.get("env", {})).items()},
        }

        try:
            with tempfile.TemporaryDirectory(prefix="qwen-") as tmp_dir:
                tmp_home = Path(tmp_dir)
                qwen_settings_path = tmp_home / "qwen-settings.json"
                qwen_settings_path.write_text(
                    json.dumps(settings, indent=2),
                    encoding="utf-8",
                )

                # Force stateless execution by isolating all runtime state to temp dirs.
                env["HOME"] = str(tmp_home)
                env["XDG_CONFIG_HOME"] = str(tmp_home / ".config")
                env["XDG_CACHE_HOME"] = str(tmp_home / ".cache")
                env["XDG_STATE_HOME"] = str(tmp_home / ".state")
                env["QWEN_CODE_SYSTEM_SETTINGS_PATH"] = str(qwen_settings_path)
                # Compatibility: older qwen-code used GEMINI_* naming.
                env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(qwen_settings_path)
                final_assistant_text: str | None = None
                pending_tool_calls: list[dict[str, Any]] = []

                def on_stdout_line(line: str) -> None:
                    nonlocal final_assistant_text
                    stripped = line.strip()
                    if not stripped:
                        return

                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        if verbosity >= 1:
                            console.print(stripped)

                        return
                    if not isinstance(event, dict):
                        if verbosity >= 1:
                            console.print(_as_json(event))

                        return

                    event_type = str(event.get("type", "")).strip().lower()
                    event_subtype = str(event.get("subtype", "")).strip().lower()
                    maybe_text = _emit_qwen_stream_event(
                        event=event,
                        console=console,
                        sink=sink,
                        verbosity=verbosity,
                        pending_tool_calls=pending_tool_calls,
                    )
                    if (
                        maybe_text
                        and event_type == "assistant"
                        and event_subtype not in {"thinking", "reasoning"}
                    ):
                        final_assistant_text = maybe_text

                run_subprocess(
                    args=[
                        binary,
                        "-p",
                        task.instruction,
                        "-y",
                        "--output-format",
                        "stream-json",
                    ],
                    cwd=str(Path.cwd()),
                    env=env,
                    check=True,
                    echo_stdout=False,
                    on_stdout_line=on_stdout_line,
                )
                for pending in pending_tool_calls:
                    display_name = str(pending.get("name") or "tool")
                    display_input = _format_tool_input(
                        arguments=pending.get("arguments")
                    )
                    if verbosity == 0:
                        console.print(
                            f"{display_name}: {_compact_single_line(display_input)}"
                        )
                        console.print("─" * max(20, console.width), style="dim")
                    else:
                        console.print(f"{display_name}: {display_input}")
                        console.print("[no detailed output]")
                        console.print("─" * max(20, console.width), style="dim")
                if final_assistant_text:
                    console.print(
                        Panel(final_assistant_text, title="Done", border_style="green")
                    )
        except FileNotFoundError:
            console.print(
                Panel(
                    "qwen CLI not found. Install @qwen-code/qwen-code and ensure `qwen` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message="qwen CLI not found",
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={
                            "kind": "missing_binary",
                            "message": "qwen CLI not found",
                        },
                    )
                )
                sink.finalize(result=result)

            return result
        except subprocess.CalledProcessError as err:
            actionable_error = _qwen_actionable_error_message(
                model=cfg.model.model,
                process_error=err,
            )
            if actionable_error:
                console.print(
                    Panel(
                        actionable_error,
                        title="Agent Compatibility Error",
                        border_style="red",
                    )
                )
                result = RunResult(
                    exit_code=1,
                    success=False,
                    task_id=task.task_id,
                    final_message=actionable_error,
                )
                if sink:
                    sink.emit(
                        event=AgentEvent(
                            event_type="issue",
                            payload={
                                "kind": "compatibility",
                                "message": actionable_error,
                            },
                        )
                    )
                    sink.finalize(result=result)

                return result

            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message=str(err),
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={"kind": "subprocess", "message": str(err)},
                    )
                )
                sink.finalize(result=result)

            return result
        except (ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message=str(err),
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={"kind": "runtime", "message": str(err)},
                    )
                )
                sink.finalize(result=result)

            return result

        result = RunResult(
            exit_code=0,
            success=True,
            task_id=task.task_id,
        )
        if sink:
            sink.emit(
                event=AgentEvent(
                    event_type="done",
                    payload={"message": "qwen run completed"},
                )
            )
            sink.finalize(result=result)

        return result
