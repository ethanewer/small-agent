from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig
from agents.liteforge.agent import (
    build_system_prompt,
    build_user_prompt,
    get_environment,
    list_cwd_files,
)
from agents.liteforge.context import Context
from agents.liteforge.logging_utils import (
    last_assistant_text,
)
from agents.liteforge.orchestrator import Orchestrator
from agents.liteforge.tools.executor import ToolExecutor
from agents.liteforge.tools.registry import (
    ALL_TOOL_NAMES,
    READONLY_TOOL_NAMES,
    build_tool_definitions,
)


def _coerce_int(*, value: Any, default: int) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_final_message_enabled(*, value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}

    return bool(value)


@contextmanager
def _temporary_environ(*, overrides: dict[str, str]) -> Any:
    original_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in original_values.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _resolve_tool_names(*, options: dict[str, Any]) -> list[str]:
    configured = options.get("tool_names")
    if isinstance(configured, list):
        tool_names = [str(name) for name in configured if str(name).strip()]
        if tool_names:
            return tool_names

    readonly = bool(options.get("readonly", False))
    if readonly:
        return list(READONLY_TOOL_NAMES)

    return list(ALL_TOOL_NAMES)


def _resolve_max_tokens(
    *,
    options: dict[str, Any],
    model_context_length: int | None,
) -> int:
    default_max_tokens = 20480
    if model_context_length is not None and model_context_length > 0:
        default_max_tokens = min(default_max_tokens, model_context_length)

    requested_max_tokens = _coerce_int(
        value=options.get("max_tokens"),
        default=default_max_tokens,
    )
    if model_context_length is not None and model_context_length > 0:
        return max(1, min(requested_max_tokens, model_context_length))

    return max(1, requested_max_tokens)


def _resolve_env(*, options: dict[str, Any]) -> dict[str, Any]:
    env = get_environment()
    configured_cwd = options.get("cwd")
    if isinstance(configured_cwd, str) and configured_cwd.strip():
        cwd_path = Path(configured_cwd)
        if cwd_path.is_absolute():
            env["cwd"] = str(cwd_path)
        else:
            env["cwd"] = str(Path.cwd() / cwd_path)

    return env


def _build_context(
    *,
    task: Task,
    options: dict[str, Any],
    cfg: AgentRuntimeConfig,
    env: dict[str, Any],
    tool_names: list[str],
) -> tuple[Context, list[dict[str, Any]]]:
    desc_context = {
        "env": env,
        "tool_names": {name: name for name in ALL_TOOL_NAMES},
        "model": {"input_modalities": []},
    }
    tool_defs = build_tool_definitions(
        tool_names=tool_names,
        description_context=desc_context,
    )

    system_parts = build_system_prompt(
        env=env,
        files=list_cwd_files(cwd=env["cwd"]),
        tool_names=tool_names,
        custom_rules=str(options.get("custom_rules", "")),
    )

    context = Context()
    context.set_system_messages(system_parts)
    context.add_user_message(
        build_user_prompt(
            event_name="user_message",
            event_value=task.instruction,
        )
    )
    context.max_tokens = _resolve_max_tokens(
        options=options,
        model_context_length=cfg.model.context_length,
    )
    context.extra_params = cfg.model.extra_params
    context.tools = tool_defs

    return context, tool_defs


def _run_agent_pass(
    *,
    context: Context,
    env: dict[str, Any],
    model: str,
    tool_names: list[str],
    options: dict[str, Any],
    stream: bool,
    console: Console | None = None,
    separator_before_stream: bool = False,
    executor: ToolExecutor | None = None,
) -> tuple[Orchestrator, bool]:
    desc_context = {
        "env": env,
        "tool_names": {name: name for name in ALL_TOOL_NAMES},
        "model": {"input_modalities": []},
    }
    tool_defs = build_tool_definitions(
        tool_names=tool_names,
        description_context=desc_context,
    )
    context.tools = tool_defs

    if executor is None:
        executor = ToolExecutor(env=env)

    max_requests_per_turn = _coerce_int(
        value=options.get("max_requests_per_turn"),
        default=100,
    )
    max_tool_failure_per_turn = _coerce_int(
        value=options.get("max_tool_failure_per_turn"),
        default=3,
    )

    orch = Orchestrator(
        context=context,
        executor=executor,
        model=model,
        tools=tool_defs,
        max_requests_per_turn=max_requests_per_turn,
        max_tool_failure_per_turn=max_tool_failure_per_turn,
        stream=stream,
    )
    if hasattr(orch, "set_log_console"):
        orch.set_log_console(console=console)
    if separator_before_stream and hasattr(orch, "queue_stream_separator"):
        orch.queue_stream_separator()

    completed = orch.run()
    return orch, completed


class LiteforgeAgent:
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
        env = _resolve_env(options=options)
        model = str(options.get("model") or cfg.model.model).strip()
        stream = bool(options.get("stream", True))
        final_message_enabled = _coerce_final_message_enabled(
            value=options.get("final_message")
        )

        env_overrides = {
            "OPENAI_MODEL": cfg.model.model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_URL": cfg.model.api_base,
            "OPENAI_API_KEY": cfg.model.api_key,
        }
        if model.lower().startswith("anthropic/"):
            env_overrides["ANTHROPIC_API_KEY"] = cfg.model.api_key

        try:
            with _temporary_environ(overrides=env_overrides):
                tool_names = _resolve_tool_names(options=options)
                context, _ = _build_context(
                    task=task,
                    options=options,
                    cfg=cfg,
                    env=env,
                    tool_names=tool_names,
                )
                _, completed = _run_agent_pass(
                    context=context,
                    env=env,
                    model=model,
                    tool_names=tool_names,
                    options=options,
                    stream=stream,
                    console=console,
                )
        except Exception as err:
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message=str(err),
            )
            if sink:
                sink.finalize(result=result)
            return result

        final_message = last_assistant_text(context=context)
        if (
            final_message_enabled
            and not stream
            and final_message
            and final_message.strip()
        ):
            console.print(
                Panel(final_message, title="Final Output", border_style="green")
            )
        exit_code = 0 if completed else 1
        result = RunResult(
            exit_code=exit_code,
            success=completed,
            task_id=task.task_id,
            final_message=final_message,
        )
        if sink:
            sink.finalize(result=result)
        return result
