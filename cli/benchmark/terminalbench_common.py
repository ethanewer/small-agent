from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.core.task import Task, TaskContext
from agents.interface import AgentModelConfig


@dataclass
class TerminalBenchSample:
    task_id: str
    instruction: str
    metadata: dict[str, Any]


def sample_to_task(*, sample: TerminalBenchSample) -> Task:
    return Task(
        task_id=sample.task_id,
        instruction=sample.instruction,
        metadata=sample.metadata,
        context=TaskContext(
            dataset="terminal-bench",
            source="terminalbench-adapter",
            metadata={"task_id": sample.task_id},
        ),
    )


def parse_terminalbench_sample(*, data: dict[str, Any]) -> TerminalBenchSample:
    task_id = str(data.get("task_id") or data.get("id") or "terminalbench-task")
    instruction = str(data.get("instruction") or data.get("prompt") or "")
    metadata = dict(data.get("metadata") or {})
    return TerminalBenchSample(
        task_id=task_id,
        instruction=instruction,
        metadata=metadata,
    )


def model_from_kwargs(
    *,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float | None = None,
) -> AgentModelConfig:
    return AgentModelConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
    )
