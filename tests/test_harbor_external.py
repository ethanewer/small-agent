from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = PROJECT_ROOT / "harbor" / "agent.py"
sys.path.insert(0, str(PROJECT_ROOT))

agent_spec = importlib.util.spec_from_file_location("harbor_agent_module", AGENT_PATH)
assert agent_spec and agent_spec.loader
harbor_agent = importlib.util.module_from_spec(agent_spec)
sys.modules["harbor_agent_module"] = harbor_agent
agent_spec.loader.exec_module(harbor_agent)
SmallAgentHarborAgent = harbor_agent.SmallAgentHarborAgent


class _FakeEnvironment:
    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result or {"exit_code": 0, "stdout": "ok", "stderr": ""}
        self._results = list(results or [])
        self.directories: set[str] = set()
        self.upload_calls: list[dict[str, str]] = []

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env or {},
                "timeout_sec": timeout_sec,
            }
        )
        if self._results:
            return self._results.pop(0)
        return self._result

    async def is_dir(self, *, path: str) -> bool:
        return path in self.directories

    async def upload_dir(self, *, source_dir: str | Path, target_dir: str) -> None:
        self.upload_calls.append(
            {
                "source_dir": str(source_dir),
                "target_dir": target_dir,
            }
        )
        self.directories.add(target_dir)


class _FakeContext:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.result: dict[str, Any] | None = None
        self.metadata: dict[str, Any] = {}

    def add_message(self, *, message: str) -> None:
        self.messages.append(message)

    def set_result(self, *, result: dict[str, Any]) -> None:
        self.result = result


class TestHarborExternalAgent(unittest.TestCase):
    def _write_config(self, payload: dict[str, Any]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with temp_file:
            json.dump(payload, temp_file)
        return Path(temp_file.name)

    def _minimal_config(self) -> dict[str, Any]:
        return {
            "default_model": "qwen3-coder-next",
            "models": {
                "qwen3-coder-next": {
                    "model": "qwen/qwen3-coder-next",
                    "api_base": "https://openrouter.ai/api/v1",
                    "api_key": "literal-key",
                },
                "gpt-5.3-codex": {
                    "model": "gpt-5.3-codex",
                    "api_base": "https://api.openai.com/v1",
                    "api_key": "literal-key",
                },
            },
            "default_agent": "terminus-2",
            "agents": {"terminus-2": {}, "qwen": {}},
            "verbosity": 0,
            "max_turns": 5,
            "max_wait_seconds": 10.0,
        }

    def test_agent_import_does_not_pull_task_runtime_modules(self) -> None:
        module_name = "harbor_agent_module_isolation"
        for module_key in ("litellm", "pexpect", module_name):
            sys.modules.pop(module_key, None)

        spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
        self.assertIsNotNone(spec)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module=module)
        self.assertNotIn("litellm", sys.modules)
        self.assertNotIn("pexpect", sys.modules)

    def test_setup_checks_cli_run_exists(self) -> None:
        environment = _FakeEnvironment()
        agent = SmallAgentHarborAgent()
        asyncio.run(agent.setup(environment=environment))
        self.assertTrue(environment.upload_calls)
        self.assertTrue(environment.calls)
        self.assertIn("/tmp/small-agent-cli/cli.py", environment.calls[0]["command"])

    def test_setup_fails_when_cli_run_missing(self) -> None:
        environment = _FakeEnvironment(
            result={
                "exit_code": 1,
                "stdout": "",
                "stderr": "small-agent cli bundle is missing required files",
            }
        )
        agent = SmallAgentHarborAgent()
        with self.assertRaises(RuntimeError):
            asyncio.run(agent.setup(environment=environment))

    def test_run_uses_default_model_and_agent(self) -> None:
        path = self._write_config(self._minimal_config())
        try:
            environment = _FakeEnvironment(
                result={"exit_code": 0, "stdout": "done", "stderr": ""}
            )
            context = _FakeContext()
            agent = SmallAgentHarborAgent(config_path=str(path))
            asyncio.run(
                agent.run(
                    instruction="echo hello",
                    environment=environment,
                    context=context,
                )
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertIsNotNone(context.result)
        assert context.result is not None
        self.assertTrue(context.result["success"])
        run_call = environment.calls[-1]
        self.assertIn("--agent terminus-2", run_call["command"])
        self.assertIn("--model qwen3-coder-next", run_call["command"])
        run_env = run_call["env"]
        self.assertEqual(run_env["OPENAI_MODEL"], "qwen/qwen3-coder-next")
        setup_metadata = context.metadata.get("small_agent_setup", {})
        self.assertEqual(
            setup_metadata.get("bootstrap_python_dependencies", {}).get("status"),
            "ok",
        )

    def test_run_honors_env_overrides(self) -> None:
        path = self._write_config(self._minimal_config())
        try:
            environment = _FakeEnvironment()
            context = _FakeContext()
            agent = SmallAgentHarborAgent(config_path=str(path))
            with patch.dict(
                "os.environ",
                {
                    "SMALL_AGENT_HARBOR_MODEL": "gpt-5.3-codex",
                    "SMALL_AGENT_HARBOR_AGENT": "qwen",
                },
                clear=False,
            ):
                asyncio.run(
                    agent.run(
                        instruction="echo hello",
                        environment=environment,
                        context=context,
                    )
                )
        finally:
            path.unlink(missing_ok=True)

        run_call = environment.calls[-1]
        self.assertIn("--agent qwen", run_call["command"])
        self.assertIn("--model gpt-5.3-codex", run_call["command"])
        self.assertEqual(context.result and context.result["exit_code"], 0)

    def test_run_populates_failure_result(self) -> None:
        path = self._write_config(self._minimal_config())
        try:
            environment = _FakeEnvironment(
                results=[
                    {"exit_code": 0, "stdout": "ok", "stderr": ""},
                    {"exit_code": 0, "stdout": "ok", "stderr": ""},
                    {"exit_code": 3, "stdout": "bad", "stderr": "failure"},
                ]
            )
            context = _FakeContext()
            agent = SmallAgentHarborAgent(config_path=str(path))
            asyncio.run(
                agent.run(
                    instruction="echo hello",
                    environment=environment,
                    context=context,
                )
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertIsNotNone(context.result)
        assert context.result is not None
        self.assertFalse(context.result["success"])
        self.assertEqual(context.result["exit_code"], 3)
        self.assertEqual(context.result["stderr"], "failure")

    def test_run_reports_unknown_agent_override(self) -> None:
        config = self._minimal_config()
        path = self._write_config(config)
        try:
            environment = _FakeEnvironment()
            context = _FakeContext()
            agent = SmallAgentHarborAgent(config_path=str(path))
            with patch.dict(
                "os.environ",
                {
                    "SMALL_AGENT_HARBOR_AGENT": "missing-agent",
                },
                clear=False,
            ):
                asyncio.run(
                    agent.run(
                        instruction="echo hello",
                        environment=environment,
                        context=context,
                    )
                )
        finally:
            path.unlink(missing_ok=True)

        self.assertIsNotNone(context.result)
        assert context.result is not None
        self.assertFalse(context.result["success"])
        self.assertIn("Unknown Harbor agent override", context.result["stderr"])


if __name__ == "__main__":
    unittest.main()
