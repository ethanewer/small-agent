from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import fields
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
    from agent_evolve.agent import ModelConfig as CoreModelConfig  # noqa: E402  # pyright: ignore[reportMissingImports]
    from agent_evolve.agent import AgentCallbacks  # noqa: E402  # pyright: ignore[reportMissingImports]
    from agent_evolve.agent import Command  # noqa: E402  # pyright: ignore[reportMissingImports]
    from agent_evolve.agent import ParsedResponse  # noqa: E402  # pyright: ignore[reportMissingImports]
    from agent_evolve.agent import run_agent  # noqa: E402  # pyright: ignore[reportMissingImports]
except ImportError:
    from agent_evolve.start_workdir import agent as agent_module  # noqa: E402
    from agent_evolve.start_workdir.agent import Agent  # noqa: E402
    from agent_evolve.start_workdir.agent import Config as CoreConfig  # noqa: E402
    from agent_evolve.start_workdir.agent import ModelConfig as CoreModelConfig  # noqa: E402
    from agent_evolve.start_workdir.agent import AgentCallbacks  # noqa: E402
    from agent_evolve.start_workdir.agent import Command  # noqa: E402, F401
    from agent_evolve.start_workdir.agent import ParsedResponse  # noqa: E402
    from agent_evolve.start_workdir.agent import run_agent  # noqa: E402

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

    # ---- wrapper-compatibility tests ----

    def test_exports_all_symbols_required_by_wrapper(self) -> None:
        """The production wrapper imports these names from core_agent."""
        required = [
            "AgentCallbacks",
            "Command",
            "Config",
            "ModelConfig",
            "ParsedResponse",
            "run_agent",
        ]
        for name in required:
            self.assertTrue(
                hasattr(agent_module, name),
                f"agent module missing export: {name}",
            )

    def test_core_config_accepts_final_message_enabled(self) -> None:
        """The wrapper passes final_message_enabled when constructing CoreConfig."""
        cfg = CoreConfig(
            active_model_key="test-model",
            active_model=CoreModelConfig(
                model="test-model",
                api_base="http://localhost",
                api_key="key",
            ),
            final_message_enabled=True,
        )
        self.assertTrue(cfg.final_message_enabled)

        cfg_disabled = CoreConfig(
            active_model_key="test-model",
            active_model=CoreModelConfig(
                model="test-model",
                api_base="http://localhost",
                api_key="key",
            ),
            final_message_enabled=False,
        )
        self.assertFalse(cfg_disabled.final_message_enabled)

    def test_core_config_has_all_required_fields(self) -> None:
        """CoreConfig must have every field the wrapper constructs."""
        field_names = {f.name for f in fields(CoreConfig)}
        required_fields = {
            "active_model_key",
            "active_model",
            "verbosity",
            "max_turns",
            "max_wait_seconds",
            "final_message_enabled",
        }
        missing = required_fields - field_names
        self.assertFalse(missing, f"CoreConfig missing fields: {missing}")

    def test_model_config_has_all_required_fields(self) -> None:
        """CoreModelConfig must accept all args the wrapper passes."""
        field_names = {f.name for f in fields(CoreModelConfig)}
        required_fields = {
            "model",
            "api_base",
            "api_key",
            "temperature",
            "context_length",
            "extra_params",
        }
        missing = required_fields - field_names
        self.assertFalse(missing, f"CoreModelConfig missing fields: {missing}")

    def test_model_config_construction_matches_wrapper(self) -> None:
        """Wrapper constructs CoreModelConfig with these exact kwargs."""
        mc = CoreModelConfig(
            model="qwen/qwen3-coder-next",
            api_base="https://openrouter.ai/api/v1",
            api_key="sk-test",
            temperature=0.7,
            context_length=262144,
        )
        self.assertEqual(mc.model, "qwen/qwen3-coder-next")
        self.assertEqual(mc.context_length, 262144)

    def test_run_agent_signature_matches_wrapper_call(self) -> None:
        """The wrapper calls run_agent with these keyword arguments."""
        sig = inspect.signature(run_agent)
        param_names = set(sig.parameters.keys())
        required_params = {"instruction", "cfg", "api_key", "callbacks"}
        missing = required_params - param_names
        self.assertFalse(missing, f"run_agent missing parameters: {missing}")

    def test_agent_callbacks_on_done_accepts_str(self) -> None:
        """The wrapper's on_done callback passes a str argument."""
        received: list[str] = []

        def on_done(text: str) -> None:
            received.append(text)

        cb = AgentCallbacks(on_done=on_done)
        assert cb.on_done is not None
        cb.on_done("Task complete.")
        self.assertEqual(received, ["Task complete."])

    def test_agent_callbacks_on_reasoning_signature(self) -> None:
        """The wrapper's on_reasoning passes (turn: int, parsed: ParsedResponse)."""
        received: list[tuple[int, ParsedResponse]] = []

        def on_reasoning(turn: int, parsed: ParsedResponse) -> None:
            received.append((turn, parsed))

        cb = AgentCallbacks(on_reasoning=on_reasoning)
        assert cb.on_reasoning is not None
        pr = ParsedResponse(
            analysis="a",
            plan="p",
            commands=[],
            task_complete=False,
        )
        cb.on_reasoning(1, pr)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], 1)

    def test_agent_callbacks_on_stopped_accepts_int(self) -> None:
        """The wrapper's on_stopped passes max_turns as int."""
        received: list[int] = []

        def on_stopped(max_turns: int) -> None:
            received.append(max_turns)

        cb = AgentCallbacks(on_stopped=on_stopped)
        assert cb.on_stopped is not None
        cb.on_stopped(50)
        self.assertEqual(received, [50])


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
