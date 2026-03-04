from __future__ import annotations

import asyncio
from typing import Any, Protocol

from rich.console import Console

from agents.core.result import RunResult
from agents.core.task import Task
from agents.interface import Agent, AgentRuntimeConfig, run_agent_task_with_fallback


class BenchmarkAgent(Protocol):
    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, Any]: ...


def result_to_payload(*, result: RunResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "success": result.success,
        "exit_code": result.exit_code,
        "final_message": result.final_message,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }


class AgentBenchmarkAdapter:
    def __init__(
        self,
        *,
        agent: Agent,
        runtime_cfg: AgentRuntimeConfig,
        console: Console | None = None,
    ) -> None:
        self._agent = agent
        self._runtime_cfg = runtime_cfg
        self._console = console or Console()

    def run_sync(self, *, task: Task) -> RunResult:
        return run_agent_task_with_fallback(
            agent=self._agent,
            task=task,
            cfg=self._runtime_cfg,
            console=self._console,
            sink=None,
        )

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, Any]:
        del extra_kwargs
        task = Task(
            task_id=str(data.get("task_id") or data.get("id") or "benchmark-task"),
            instruction=str(data.get("instruction") or data.get("prompt") or ""),
            metadata=dict(data.get("metadata") or {}),
        )
        result = await asyncio.to_thread(self.run_sync, task=task)
        return result_to_payload(result=result)
