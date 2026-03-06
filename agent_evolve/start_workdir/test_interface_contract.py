from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agent_evolve import agent as agent_module  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]
    from agent_evolve.agent import Agent  # noqa: E402  # pyright: ignore[reportMissingImports]
    from agent_evolve.agent import Config as CoreConfig  # noqa: E402  # pyright: ignore[reportMissingImports]
except ImportError:
    from agent_evolve.start_workdir import agent as agent_module  # noqa: E402
    from agent_evolve.start_workdir.agent import Agent  # noqa: E402
    from agent_evolve.start_workdir.agent import Config as CoreConfig  # noqa: E402

from agents.core.result import RunResult  # noqa: E402
from agents.core.task import Task  # noqa: E402
from agents.interface import (  # noqa: E402
    AgentModelConfig,
    AgentRuntimeConfig,
    run_agent_task_with_fallback,
)


class TestAgentContract(unittest.TestCase):
    def _runtime_cfg(self) -> AgentRuntimeConfig:
        return AgentRuntimeConfig(
            agent_key="evolver-workdir",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={"max_turns": 3, "max_wait_seconds": 2.0},
        )

    def test_run_returns_exit_code_from_run_task(self) -> None:
        runtime = self._runtime_cfg()
        with patch.object(agent_module, "run_agent", return_value=0):
            exit_code = Agent().run(
                instruction="echo hi",
                cfg=runtime,
                console=Console(record=True),
            )
        self.assertEqual(exit_code, 0)

    def test_run_task_returns_run_result(self) -> None:
        runtime = self._runtime_cfg()
        with patch.object(agent_module, "run_agent", return_value=0):
            result = Agent().run_task(
                task=Task.from_instruction(
                    instruction="inspect",
                    task_id="wk-1",
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "wk-1")
        self.assertEqual(result.exit_code, 0)

    def test_maps_runtime_config_to_core_config(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="evolver-workdir",
            model=AgentModelConfig(
                model="gpt-5.3-codex",
                api_base="https://api.openai.com/v1",
                api_key="test-key",
                temperature=0.1,
            ),
            agent_config={"verbosity": 0, "max_turns": 8, "max_wait_seconds": 7.5},
        )
        captured_kwargs: dict[str, object] = {}

        def fake_run_agent(**kwargs: object) -> int:
            captured_kwargs.update(kwargs)
            return 0

        with patch.object(agent_module, "run_agent", side_effect=fake_run_agent):
            result = Agent().run_task(
                task=Task.from_instruction(
                    instruction="inspect config",
                    task_id="wk-2",
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertTrue(result.success)
        cfg = cast(CoreConfig, captured_kwargs["cfg"])
        self.assertEqual(cfg.max_turns, 8)
        self.assertEqual(cfg.max_wait_seconds, 7.5)
        self.assertEqual(cfg.active_model.model, "gpt-5.3-codex")

    def test_fallback_runner_uses_run_task_for_benchmark_compatibility(self) -> None:
        runtime = self._runtime_cfg()
        task = Task.from_instruction(instruction="from benchmark", task_id="bm-1")
        with patch.object(agent_module, "run_agent", return_value=0):
            result: RunResult = run_agent_task_with_fallback(
                agent=Agent(),
                task=task,
                cfg=runtime,
                console=Console(record=True),
            )
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "bm-1")


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
