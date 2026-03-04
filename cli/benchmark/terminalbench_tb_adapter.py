from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.adapters import AgentBenchmarkAdapter, result_to_payload
from benchmark.terminalbench_common import parse_terminalbench_sample, sample_to_task


class TerminalBenchTBAdapter:
    def __init__(self, *, adapter: AgentBenchmarkAdapter) -> None:
        self._adapter = adapter

    def run_sample(self, *, sample_data: dict[str, Any]) -> dict[str, Any]:
        sample = parse_terminalbench_sample(data=sample_data)
        task = sample_to_task(sample=sample)
        result = self._adapter.run_sync(task=task)
        payload = result_to_payload(result=result)
        payload["dataset"] = "terminal-bench"
        return payload

    def run_samples_to_jsonl(
        self,
        *,
        samples: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                payload = self.run_sample(sample_data=sample)
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
