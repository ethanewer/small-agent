from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from rich.console import Console


@dataclass
class AgentModelConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float | None = None
    context_length: int | None = None
    extra_params: dict[str, Any] | None = None


@dataclass
class AgentRuntimeConfig:
    agent_key: str
    model: AgentModelConfig
    agent_config: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    def run(
        self,
        instruction: str,
        cfg: AgentRuntimeConfig,
        console: Console,
    ) -> int: ...


class TaskAgent(Protocol):
    def run_task(
        self,
        *,
        task: Task,
        cfg: AgentRuntimeConfig,
        console: Console | None = None,
        sink: EventSink | None = None,
    ) -> RunResult: ...


def run_agent_task_with_fallback(
    *,
    agent: Agent,
    task: Task,
    cfg: AgentRuntimeConfig,
    console: Console,
    sink: EventSink | None = None,
) -> RunResult:
    runner = agent
    if hasattr(runner, "run_task"):
        return getattr(runner, "run_task")(  # noqa: B009
            task=task,
            cfg=cfg,
            console=console,
            sink=sink,
        )

    exit_code = agent.run(
        instruction=task.instruction,
        cfg=cfg,
        console=console,
    )
    return RunResult(
        exit_code=exit_code,
        success=exit_code == 0,
        task_id=task.task_id,
    )
