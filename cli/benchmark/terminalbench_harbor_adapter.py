from __future__ import annotations

from typing import Any

from benchmark.adapters import AgentBenchmarkAdapter
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
        result = self._adapter.run_sync(task=task)
        return {
            "task_id": result.task_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "metrics": result.metrics,
        }
