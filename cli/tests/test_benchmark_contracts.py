from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.core.result import RunResult  # noqa: E402
from agents.interface import AgentModelConfig, AgentRuntimeConfig  # noqa: E402
from agents.registry import get_agent  # noqa: E402
from benchmark.adapters import AgentBenchmarkAdapter, result_to_payload  # noqa: E402
from benchmark.areal_adapter import AReaLAdapter  # noqa: E402
from benchmark.terminalbench_common import (  # noqa: E402
    parse_terminalbench_sample,
    sample_to_task,
)
from benchmark.terminalbench_harbor_adapter import HarborImportPathAgent  # noqa: E402
from benchmark.terminalbench_tb_adapter import TerminalBenchTBAdapter  # noqa: E402


def _runtime_cfg() -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        agent_key="terminus-2",
        model=AgentModelConfig(
            model="qwen/qwen3-coder-next",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
        ),
        agent_config={"final_message": False},
    )


class TestBenchmarkAdapters(unittest.TestCase):
    def test_result_to_payload_preserves_metrics_and_artifacts(self) -> None:
        result = RunResult(
            exit_code=0,
            success=True,
            task_id="task-1",
            final_message="done",
            metrics={"resolved": 1.0},
            artifacts={"trace": "/tmp/trace.jsonl"},
        )
        payload = result_to_payload(result=result)
        self.assertEqual(payload["task_id"], "task-1")
        self.assertEqual(payload["metrics"], {"resolved": 1.0})
        self.assertEqual(payload["artifacts"], {"trace": "/tmp/trace.jsonl"})
        self.assertEqual(payload["final_message"], "done")

    def test_agent_benchmark_adapter_async_run_payload_shape(self) -> None:
        with patch(
            "agents.terminus2.agent.run_agent",
            return_value=0,
        ):
            adapter = AgentBenchmarkAdapter(
                agent=get_agent("terminus-2"),
                runtime_cfg=_runtime_cfg(),
                console=Console(record=True),
            )
            payload = asyncio.run(
                adapter.run(
                    data={"task_id": "tb-7", "instruction": "echo ok", "metadata": {}}
                )
            )
        self.assertEqual(payload["task_id"], "tb-7")
        self.assertEqual(payload["exit_code"], 0)
        self.assertTrue(payload["success"])

    def test_terminalbench_parsing_and_task_context_fallbacks(self) -> None:
        sample = parse_terminalbench_sample(
            data={"id": "fallback-id", "prompt": "Solve this", "metadata": {"a": 1}}
        )
        task = sample_to_task(sample=sample)
        self.assertEqual(sample.task_id, "fallback-id")
        self.assertEqual(sample.instruction, "Solve this")
        self.assertEqual(task.context.dataset, "terminal-bench")
        self.assertEqual(task.context.metadata["task_id"], "fallback-id")

    def test_terminalbench_tb_adapter_writes_jsonl(self) -> None:
        with patch(
            "agents.terminus2.agent.run_agent",
            return_value=0,
        ):
            adapter = AgentBenchmarkAdapter(
                agent=get_agent("terminus-2"),
                runtime_cfg=_runtime_cfg(),
                console=Console(record=True),
            )
            tb = TerminalBenchTBAdapter(adapter=adapter)
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "tb.jsonl"
                tb.run_samples_to_jsonl(
                    samples=[{"task_id": "tb-1", "instruction": "echo hello"}],
                    output_path=output_path,
                )
                lines = output_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
        self.assertEqual(payload["task_id"], "tb-1")
        self.assertEqual(payload["dataset"], "terminal-bench")
        self.assertTrue(payload["success"])

    def test_harbor_import_path_agent_contract(self) -> None:
        with patch(
            "agents.terminus2.agent.run_agent",
            return_value=0,
        ):
            adapter = AgentBenchmarkAdapter(
                agent=get_agent("terminus-2"),
                runtime_cfg=_runtime_cfg(),
                console=Console(record=True),
            )
            harbor = HarborImportPathAgent(adapter=adapter)
            payload = asyncio.run(
                harbor.run(
                    data={
                        "task_id": "harbor-1",
                        "instruction": "echo hello",
                        "metadata": {"suite": "harbor"},
                    }
                )
            )
        self.assertEqual(payload["task_id"], "harbor-1")
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("metrics", payload)

    def test_areal_adapter_reward_mapping_with_mock(self) -> None:
        adapter = MagicMock()
        adapter.run_sync.return_value = RunResult(
            exit_code=0, success=True, task_id="a-1"
        )
        areal = AReaLAdapter(adapter=adapter)
        reward = asyncio.run(
            areal.run(
                data={"id": "a-1", "messages": [{"role": "user", "content": "x"}]}
            )
        )
        self.assertEqual(reward, 1.0)

        adapter.run_sync.return_value = RunResult(
            exit_code=1, success=False, task_id="a-2"
        )
        reward_fail = asyncio.run(areal.run(data={"id": "a-2", "prompt": "x"}))
        self.assertEqual(reward_fail, 0.0)


if __name__ == "__main__":
    unittest.main()
