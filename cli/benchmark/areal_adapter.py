from __future__ import annotations

import asyncio
from typing import Any

from benchmark.adapters import AgentBenchmarkAdapter
from agents.core.task import Task, TaskContext


class AReaLAdapter:
    """
    AReaL-compatible async workflow wrapper.

    Signature matches the tutorial pattern: run(data, **extra_kwargs).
    """

    def __init__(self, *, adapter: AgentBenchmarkAdapter) -> None:
        self._adapter = adapter

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        # Caller may pass these values via AReaL workflow kwargs.
        _ = extra_kwargs.get("http_client")
        _ = extra_kwargs.get("base_url")
        _ = extra_kwargs.get("api_key")

        instruction = str(
            data.get("instruction")
            or data.get("prompt")
            or (data.get("messages") or [{}])[-1].get("content", "")
        )
        task = Task(
            task_id=str(data.get("id") or data.get("task_id") or "areal-task"),
            instruction=instruction,
            messages=list(data.get("messages") or []),
            metadata=dict(data.get("metadata") or {}),
            context=TaskContext(dataset="areal", source="areal-adapter"),
        )
        result = await asyncio.to_thread(self._adapter.run_sync, task=task)
        reward = 1.0 if result.success else 0.0
        return reward
