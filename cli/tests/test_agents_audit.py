from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from typing import cast
from unittest.mock import patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.core.events import AgentEvent  # noqa: E402
from agents.core.result import RunResult  # noqa: E402
from agents.core.sink import EventSink, JsonlEventSink  # noqa: E402
from agents.core.task import Task  # noqa: E402
from agents.interface import (  # noqa: E402
    AgentModelConfig,
    AgentRuntimeConfig,
    run_agent_task_with_fallback,
)
from agents.qwen.qwen_agent import QwenHeadlessAgent  # noqa: E402
from agents.registry import get_agent  # noqa: E402
from agents.terminus2 import agent as terminus_agent  # noqa: E402


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.result: RunResult | None = None

    def emit(self, *, event: AgentEvent) -> None:
        self.events.append(event)

    def finalize(self, *, result: RunResult) -> None:
        self.result = result


class _LegacyOnlyAgent:
    def __init__(self, *, exit_code: int) -> None:
        self._exit_code = exit_code
        self.calls: list[str] = []

    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        del cfg, console
        self.calls.append(instruction)
        return self._exit_code


class TestRegistryAndCore(unittest.TestCase):
    def test_get_agent_unknown_lists_available_agents(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available agents: qwen, terminus-2",
        ):
            get_agent("missing-agent")

    def test_jsonl_sink_writes_events_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "events.jsonl"
            sink = JsonlEventSink(output_path=path)
            sink.emit(
                event=AgentEvent(
                    event_type="issue",
                    payload={"kind": "parser", "message": "bad json"},
                    turn=2,
                )
            )
            sink.finalize(
                result=RunResult(
                    exit_code=0,
                    success=True,
                    task_id="t-1",
                )
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn('"event_type": "issue"', text)
            self.assertIn('"event_type": "result"', text)


class TestFallbackRunnerContracts(unittest.TestCase):
    def _runtime_cfg(self) -> AgentRuntimeConfig:
        return AgentRuntimeConfig(
            agent_key="terminus-2",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={"final_message": False},
        )

    def test_fallback_uses_run_task_when_available(self) -> None:
        runtime = self._runtime_cfg()
        task = Task.from_instruction(instruction="echo hi", task_id="task-1")
        with patch("agents.terminus2.agent.run_agent", return_value=0):
            result = run_agent_task_with_fallback(
                agent=get_agent("terminus-2"),
                task=task,
                cfg=runtime,
                console=Console(record=True),
            )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.task_id, "task-1")

    def test_fallback_uses_legacy_run_when_no_run_task(self) -> None:
        runtime = self._runtime_cfg()
        task = Task.from_instruction(instruction="legacy call", task_id="legacy-1")
        agent = _LegacyOnlyAgent(exit_code=2)
        result = run_agent_task_with_fallback(
            agent=agent,  # type: ignore[arg-type]
            task=task,
            cfg=runtime,
            console=Console(record=True),
        )
        self.assertEqual(agent.calls, ["legacy call"])
        self.assertEqual(result.exit_code, 2)
        self.assertFalse(result.success)
        self.assertEqual(result.task_id, "legacy-1")


class TestQwenAndTerminusTaskAPI(unittest.TestCase):
    def test_qwen_task_path_uses_task_instruction(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={"binary": "qwen-custom"},
        )
        captured: dict[str, object] = {}

        def fake_run_subprocess(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 0

        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=fake_run_subprocess,
        ):
            result = QwenHeadlessAgent().run_task(
                task=Task.from_instruction(
                    instruction="echo from task",
                    task_id="task-123",
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "task-123")
        self.assertEqual(
            cast(list[str], captured["args"]),
            ["qwen-custom", "-p", "echo from task", "-y"],
        )

    def test_qwen_returns_error_when_binary_missing(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={"binary": "qwen-custom"},
        )
        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=FileNotFoundError(),
        ):
            result = QwenHeadlessAgent().run_task(
                task=Task.from_instruction(
                    instruction="echo from task", task_id="task-404"
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("not found", str(result.final_message))

    def test_qwen_returns_error_on_called_process_error(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={"binary": "qwen-custom"},
        )
        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=__import__("subprocess").CalledProcessError(
                returncode=2,
                cmd=["qwen-custom"],
                stderr="boom",
            ),
        ):
            result = QwenHeadlessAgent().run_task(
                task=Task.from_instruction(
                    instruction="echo from task", task_id="task-500"
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 1)
        self.assertIsNotNone(result.final_message)

    def test_qwen_returns_compatibility_error_when_preflight_fails(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
        )
        with patch(
            "agents.qwen.qwen_agent.preflight_agent_model_compatibility",
            return_value="compatibility mismatch",
        ):
            result = QwenHeadlessAgent().run_task(
                task=Task.from_instruction(
                    instruction="echo from task", task_id="task-compat"
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.final_message, "compatibility mismatch")

    def test_terminus_run_task_returns_run_result(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="terminus-2",
            model=AgentModelConfig(
                model="model-y",
                api_base="https://example.invalid/v1",
                api_key="api",
            ),
            agent_config={"max_turns": 1, "max_wait_seconds": 1.0},
        )
        with patch("agents.terminus2.agent.run_agent", return_value=0):
            result = terminus_agent.Terminus2Agent().run_task(
                task=Task.from_instruction(instruction="inspect", task_id="t2"),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "t2")

    def test_terminus_run_task_maps_core_config_and_disable_final_message(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="terminus-2",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                temperature=0.0,
            ),
            agent_config={
                "verbosity": 3,
                "max_turns": 9,
                "max_wait_seconds": 7.5,
                "final_message": False,
            },
        )
        captured_kwargs: dict[str, object] = {}

        def fake_run_agent(**kwargs: object) -> int:
            captured_kwargs.update(kwargs)
            return 0

        with patch("agents.terminus2.agent.run_agent", side_effect=fake_run_agent):
            result = terminus_agent.Terminus2Agent().run_task(
                task=Task.from_instruction(
                    instruction="inspect config",
                    task_id="term-cfg",
                ),
                cfg=runtime,
                console=Console(record=True),
                sink=None,
            )
        self.assertTrue(result.success)
        cfg = cast(terminus_agent.CoreConfig, captured_kwargs["cfg"])
        self.assertEqual(cfg.max_turns, 9)
        self.assertEqual(cfg.max_wait_seconds, 7.5)
        self.assertFalse(cfg.final_message_enabled)
        self.assertEqual(cfg.active_model.model, "qwen/qwen3-coder-next")

    def test_terminus_emits_sink_events_from_callbacks(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="terminus-2",
            model=AgentModelConfig(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={
                "verbosity": 1,
                "max_turns": 2,
                "max_wait_seconds": 2.0,
                "final_message": False,
            },
        )
        sink = _RecordingSink()

        def fake_run_agent(**kwargs: object) -> int:
            callbacks = cast(terminus_agent.AgentCallbacks, kwargs["callbacks"])
            parsed = terminus_agent.ParsedResponse(
                analysis="a",
                plan="p",
                commands=[],
                task_complete=True,
                final_message=None,
            )
            command = terminus_agent.Command(keystrokes="echo hi\n", duration=0.1)
            if callbacks.on_reasoning:
                callbacks.on_reasoning(1, parsed)
            if callbacks.on_command_output:
                callbacks.on_command_output(command, "hi")
            if callbacks.on_issue:
                callbacks.on_issue("model", "rate limit")
            if callbacks.on_done:
                callbacks.on_done("done")
            if callbacks.on_stopped:
                callbacks.on_stopped(2)
            return 0

        with patch("agents.terminus2.agent.run_agent", side_effect=fake_run_agent):
            result = terminus_agent.Terminus2Agent().run_task(
                task=Task.from_instruction(instruction="emit events", task_id="evt-1"),
                cfg=runtime,
                console=Console(record=True),
                sink=sink,
            )
        self.assertTrue(result.success)
        self.assertIsNotNone(sink.result)
        event_types = [event.event_type for event in sink.events]
        self.assertIn("reasoning", event_types)
        self.assertIn("command_output", event_types)
        self.assertIn("issue", event_types)
        self.assertIn("done", event_types)
        self.assertIn("stopped", event_types)


if __name__ == "__main__":
    unittest.main()
