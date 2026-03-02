from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.interface import AgentRuntimeConfig
from agents.openai_compat import normalize_openai_compatible_model
from agents.toolmind_harness.harness import HarnessCallbacks, ToolCall, run_harness


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True

        if normalized in {"0", "false", "no", "off"}:
            return False

    if value is None:
        return default

    return bool(value)


def _tool_call_line(call: ToolCall) -> str:
    return f"{call.server_name}.{call.tool_name}"


def _tool_result_preview(result: dict[str, Any]) -> str:
    compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(compact) > 200:
        return compact[:197] + "..."

    return compact


class ToolmindAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        normalized_model = normalize_openai_compatible_model(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        verbosity = int(options.get("verbosity", 1))
        max_turns = int(
            options.get("max_assistant_turns", options.get("max_turns", 50))
        )
        default_temperature = (
            cfg.model.temperature if cfg.model.temperature is not None else 0.2
        )
        temperature = float(options.get("temperature", default_temperature))
        strict_protocol = _coerce_bool(options.get("strict_protocol"), default=True)
        min_tool_turns = int(options.get("min_tool_turns", 8))
        repair_attempts = int(options.get("repair_attempts", 3))
        allow_fallback_search = _coerce_bool(
            options.get("allow_fallback_search"),
            default=False,
        )
        force_think_tag = _coerce_bool(options.get("force_think_tag"), default=False)
        request_reasoning = _coerce_bool(options.get("request_reasoning"), default=True)
        internal_protocol_retry = _coerce_bool(
            options.get("internal_protocol_retry"),
            default=True,
        )
        max_internal_protocol_retries = int(
            options.get("max_internal_protocol_retries", 2)
        )
        record_protocol_repairs = _coerce_bool(
            options.get("record_protocol_repairs"),
            default=False,
        )
        key = str(options.get("key", "toolmind-local"))
        row_id = str(options.get("id", "cli-run"))

        with tempfile.TemporaryDirectory(prefix="toolmind-run-") as temp_dir:
            output_path = Path(temp_dir) / "trajectory.json"
            callbacks = HarnessCallbacks(
                on_reasoning=lambda turn, reasoning, _content: (
                    console.print(
                        Panel(
                            reasoning or "(no reasoning)",
                            title=f"Turn {turn} Reasoning",
                            border_style="magenta",
                        )
                    )
                    if verbosity >= 3
                    else None
                ),
                on_tool_call=lambda _turn, call: console.print(
                    Text(f"tool: {_tool_call_line(call)}", style="cyan")
                ),
                on_tool_result=lambda _turn, call, result: (
                    console.print(
                        Text(
                            f"result: {_tool_call_line(call)} -> "
                            f"{_tool_result_preview(result)}",
                            style="green",
                        )
                    )
                    if verbosity >= 1
                    else None
                ),
                on_issue=lambda kind, message: console.print(
                    Text(f"error: {kind} {message}", style="red")
                ),
                on_done=lambda done_text: console.print(
                    Panel(done_text, title="Done", border_style="green")
                ),
                on_stopped=lambda limit: console.print(
                    Panel(
                        f"Reached max turns ({limit}) without completion.",
                        title="Stopped",
                        border_style="yellow",
                    )
                ),
            )
            try:
                run_harness(
                    question=instruction,
                    model=normalized_model,
                    output_path=output_path,
                    key=key,
                    row_id=row_id,
                    max_assistant_turns=max_turns,
                    temperature=temperature,
                    strict_protocol=strict_protocol,
                    min_tool_turns=min_tool_turns,
                    repair_attempts=repair_attempts,
                    allow_fallback_search=allow_fallback_search,
                    force_think_tag=force_think_tag,
                    request_reasoning=request_reasoning,
                    internal_protocol_retry=internal_protocol_retry,
                    max_internal_protocol_retries=max_internal_protocol_retries,
                    record_protocol_repairs=record_protocol_repairs,
                    api_key=cfg.model.api_key,
                    api_base=cfg.model.api_base,
                    callbacks=callbacks,
                )
            except Exception as err:
                console.print(Panel(str(err), title="Agent Error", border_style="red"))
                return 1
        return 0
