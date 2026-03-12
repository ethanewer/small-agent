from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

from rich.console import Console

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


def _coerce_float(*, value: Any, default: float) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _last_assistant_text(*, context: Context) -> str | None:
    for message in reversed(context.messages):
        if message.role == "assistant" and message.content:
            content = message.content.strip()
            if content:
                return content
    return None


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
        del console

        options = cfg.agent_config
        env = get_environment()
        configured_cwd = options.get("cwd")
        if isinstance(configured_cwd, str) and configured_cwd.strip():
            cwd_path = Path(configured_cwd)
            if cwd_path.is_absolute():
                env["cwd"] = str(cwd_path)
            else:
                env["cwd"] = str(Path.cwd() / cwd_path)

        tool_names = _resolve_tool_names(options=options)
        desc_context = {
            "env": env,
            "tool_names": {name: name for name in ALL_TOOL_NAMES},
            "model": {"input_modalities": []},
        }
        tool_defs = build_tool_definitions(
            tool_names=tool_names,
            description_context=desc_context,
        )

        context = Context()
        context.set_system_messages(
            build_system_prompt(
                env=env,
                files=list_cwd_files(cwd=env["cwd"]),
                tool_names=tool_names,
                custom_rules=str(options.get("custom_rules", "")),
            )
        )
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
        context.temperature = cfg.model.temperature
        context.top_p = _coerce_float(value=options.get("top_p"), default=0.8)
        context.top_k = _coerce_int(value=options.get("top_k"), default=30)
        context.tools = tool_defs

        model = str(options.get("model") or cfg.model.model).strip()
        max_requests_per_turn = _coerce_int(
            value=options.get("max_requests_per_turn"),
            default=100,
        )
        max_tool_failure_per_turn = _coerce_int(
            value=options.get("max_tool_failure_per_turn"),
            default=3,
        )
        stream = bool(options.get("stream", True))

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
                orch = Orchestrator(
                    context=context,
                    executor=ToolExecutor(env=env),
                    model=model,
                    tools=tool_defs,
                    max_requests_per_turn=max_requests_per_turn,
                    max_tool_failure_per_turn=max_tool_failure_per_turn,
                    stream=stream,
                )
                completed = orch.run()
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

        final_message = _last_assistant_text(context=context)
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
