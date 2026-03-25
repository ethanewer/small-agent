from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents  # noqa: E402
from agents.agent_types import Config, Logger, RunResult  # noqa: E402


def _test_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "model": "model-y",
        "api_base": "https://example.invalid/v1",
        "api_key": "test-key",
        "max_turns": 2,
        "max_wait_seconds": 1.0,
    }
    defaults.update(overrides)
    return Config(**defaults)


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        turn: int | None = None,
    ) -> None:
        self.events.append({"event_type": event_type, "payload": payload, "turn": turn})


class TestRunFunction(unittest.TestCase):
    def test_run_returns_run_result(self) -> None:
        config = _test_config()
        with patch.object(
            agents,
            "run",
            return_value=RunResult(exit_code=0, success=True),
        ):
            result = agents.run(instruction="inspect", config=config)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.success)

    def test_run_passes_config_directly(self) -> None:
        config = _test_config(
            model="qwen/qwen3-coder-next",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
            temperature=0.0,
            max_turns=9,
            max_wait_seconds=7.5,
        )
        captured_kwargs: dict[str, object] = {}

        def fake_run_agent(**kwargs: object) -> RunResult:
            captured_kwargs.update(kwargs)
            return RunResult(exit_code=0, success=True)

        with patch.object(agents, "run", side_effect=fake_run_agent):
            result = agents.run(instruction="inspect config", config=config)

        self.assertTrue(result.success)
        cfg = cast(Config, captured_kwargs["config"])
        self.assertEqual(cfg.max_turns, 9)
        self.assertEqual(cfg.max_wait_seconds, 7.5)
        self.assertEqual(cfg.model, "qwen/qwen3-coder-next")
        self.assertEqual(cfg.api_key, "test-key")

    def test_run_passes_logger_to_core(self) -> None:
        config = _test_config()
        logger = _RecordingLogger()
        captured_kwargs: dict[str, object] = {}

        def fake_run_agent(**kwargs: object) -> RunResult:
            captured_kwargs.update(kwargs)
            log = cast(Logger, kwargs["logger"])
            log.log(
                event_type="reasoning",
                payload={"analysis": "a", "plan": "p"},
                turn=1,
            )
            log.log(
                event_type="command_output",
                payload={"keystrokes": "echo hi\n", "duration": 0.1, "output": "hi"},
            )
            log.log(
                event_type="issue",
                payload={"kind": "model", "message": "rate limit"},
            )
            log.log(event_type="done", payload={"message": "done"})
            log.log(event_type="stopped", payload={"max_turns": 2})
            return RunResult(exit_code=0, success=True)

        with patch.object(agents, "run", side_effect=fake_run_agent):
            result = agents.run(
                instruction="emit events",
                config=config,
                logger=logger,  # pyright: ignore[reportArgumentType]
            )

        self.assertTrue(result.success)
        self.assertIs(captured_kwargs["logger"], logger)
        event_types = [e["event_type"] for e in logger.events]
        self.assertIn("reasoning", event_types)
        self.assertIn("command_output", event_types)
        self.assertIn("issue", event_types)
        self.assertIn("done", event_types)
        self.assertIn("stopped", event_types)


if __name__ == "__main__":
    unittest.main()
