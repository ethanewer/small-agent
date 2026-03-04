from __future__ import annotations

import asyncio
from typing import Any

from benchmark.adapters import AgentBenchmarkAdapter, result_to_payload
from benchmark.terminalbench_common import parse_terminalbench_sample, sample_to_task


class HarborImportPathAgent:
    """
    Harbor-compatible adapter surface.

    Harbor can import this class by module path and call `run(data=...)`.
    """

    def __init__(self, *, adapter: AgentBenchmarkAdapter) -> None:
        self._adapter = adapter

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, Any]:
        del extra_kwargs
        sample = parse_terminalbench_sample(data=data)
        task = sample_to_task(sample=sample)
        result = await asyncio.to_thread(self._adapter.run_sync, task=task)
        payload = result_to_payload(result=result)
        return {
            "task_id": payload["task_id"],
            "success": payload["success"],
            "exit_code": payload["exit_code"],
            "metrics": payload["metrics"],
        }
