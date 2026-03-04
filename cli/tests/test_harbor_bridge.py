from __future__ import annotations

import unittest
from pathlib import Path
import sys
from typing import Any, cast
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.core.result import RunResult  # noqa: E402
from benchmark.harbor_bridge import (  # noqa: E402
    HarborTB2DefaultAgent,
    resolve_harbor_config,
)
from cli import ConfigModelEntry, LoadedConfig  # noqa: E402


def _loaded_config() -> LoadedConfig:
    return LoadedConfig(
        default_model="qwen3-coder-next",
        models={
            "qwen3-coder-next": ConfigModelEntry(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="OPENROUTER_API_KEY",
                temperature=None,
            )
        },
        default_agent="terminus-2",
        agents={"terminus-2": {"final_message": False}},
        verbosity=0,
        max_turns=50,
        max_wait_seconds=60.0,
    )


class TestHarborBridge(unittest.TestCase):
    def test_resolve_harbor_config_uses_defaults(self) -> None:
        cfg = _loaded_config()
        with patch("benchmark.harbor_bridge.cli_module.load_config", return_value=cfg):
            resolved = resolve_harbor_config()

        self.assertEqual(resolved.config_path.name, "config.json")
        self.assertEqual(resolved.agent_key, "terminus-2")
        self.assertEqual(resolved.model_key, "qwen3-coder-next")

    def test_resolve_harbor_config_supports_overrides(self) -> None:
        cfg = _loaded_config()
        with patch("benchmark.harbor_bridge.cli_module.load_config", return_value=cfg):
            resolved = resolve_harbor_config(
                config_path="/tmp/custom.json",
                agent_key="qwen",
                model_key="gpt-5.3-codex",
            )

        self.assertEqual(resolved.config_path, Path("/tmp/custom.json"))
        self.assertEqual(resolved.agent_key, "qwen")
        self.assertEqual(resolved.model_key, "gpt-5.3-codex")

    def test_perform_task_maps_success_result(self) -> None:
        run_result = RunResult(
            exit_code=0,
            success=True,
            task_id="task-1",
            metrics={
                "input_tokens": 120.0,
                "output_tokens": 45.0,
            },
        )
        fake_adapter = type(
            "FakeAdapter",
            (),
            {"run_sync": lambda self, task: run_result},
        )()
        with (
            patch(
                "benchmark.harbor_bridge.resolve_harbor_config",
                return_value=type(
                    "Resolved",
                    (),
                    {
                        "config_path": Path("/tmp/config.json"),
                        "agent_key": "terminus-2",
                        "model_key": "qwen3-coder-next",
                        "loaded": _loaded_config(),
                    },
                )(),
            ),
            patch(
                "benchmark.harbor_bridge.build_adapter_from_config",
                return_value=fake_adapter,
            ),
        ):
            agent = HarborTB2DefaultAgent()
            agent_result = agent.perform_task(
                instruction="echo hi",
                session=cast(Any, object()),
            )

        self.assertEqual(agent_result.total_input_tokens, 120)
        self.assertEqual(agent_result.total_output_tokens, 45)
        self.assertIn("none", str(agent_result.failure_mode))

    def test_perform_task_maps_failure_result(self) -> None:
        run_result = RunResult(
            exit_code=1,
            success=False,
            task_id="task-2",
            metrics={"prompt_tokens": 30.0, "completion_tokens": 10.0},
        )
        fake_adapter = type(
            "FakeAdapter",
            (),
            {"run_sync": lambda self, task: run_result},
        )()
        with (
            patch(
                "benchmark.harbor_bridge.resolve_harbor_config",
                return_value=type(
                    "Resolved",
                    (),
                    {
                        "config_path": Path("/tmp/config.json"),
                        "agent_key": "terminus-2",
                        "model_key": "qwen3-coder-next",
                        "loaded": _loaded_config(),
                    },
                )(),
            ),
            patch(
                "benchmark.harbor_bridge.build_adapter_from_config",
                return_value=fake_adapter,
            ),
        ):
            agent = HarborTB2DefaultAgent()
            agent_result = agent.perform_task(
                instruction="echo hi",
                session=cast(Any, object()),
            )

        self.assertEqual(agent_result.total_input_tokens, 30)
        self.assertEqual(agent_result.total_output_tokens, 10)
        self.assertIn("unknown_agent_error", str(agent_result.failure_mode))


if __name__ == "__main__":
    unittest.main()
