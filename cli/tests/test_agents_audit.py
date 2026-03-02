from __future__ import annotations

import tempfile
import subprocess
import unittest
from pathlib import Path
import sys
from typing import cast
from unittest.mock import patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.claude.claude_agent import ClaudeCodeAgent  # noqa: E402
from agents.interface import AgentModelConfig, AgentRuntimeConfig  # noqa: E402
from agents.openai_compat import (  # noqa: E402
    detect_provider_kind,
    preflight_agent_model_compatibility,
)
from agents.qwen.qwen_agent import QwenHeadlessAgent  # noqa: E402
from agents.registry import get_agent  # noqa: E402
from agents.terminus2 import agent as terminus_agent  # noqa: E402
from agents.toolmind_harness import harness  # noqa: E402
from agents.toolmind_harness.agent import ToolmindAgent  # noqa: E402


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._payload = text.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        del exc_type, exc, tb


class _FakeClient:
    def __init__(self, responses: list[harness.CompletionResult]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, messages, temperature=0.2, include_reasoning=True):  # type: ignore[no-untyped-def]
        del messages, temperature, include_reasoning
        idx = self.calls
        self.calls += 1
        return self._responses[idx]


class _FakeTools:
    def __init__(self) -> None:
        self.calls: list[harness.ToolCall] = []

    def execute(self, call: harness.ToolCall) -> dict[str, object]:
        self.calls.append(call)
        return {"success": True, "tool": f"{call.server_name}.{call.tool_name}"}


class TestToolmindAgentConfigCoercion(unittest.TestCase):
    def test_run_coerces_string_booleans_and_preserves_zero_temperature(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="toolmind-harness",
            model=AgentModelConfig(
                model="model-x",
                api_base="https://example.invalid/v1",
                api_key="k",
                temperature=0.0,
            ),
            agent_config={
                "strict_protocol": "false",
                "allow_fallback_search": "true",
                "force_think_tag": "false",
                "request_reasoning": "false",
                "internal_protocol_retry": "0",
                "record_protocol_repairs": "1",
            },
        )
        captured_kwargs: dict[str, object] = {}

        def fake_run_harness(**kwargs):  # type: ignore[no-untyped-def]
            captured_kwargs.update(kwargs)
            return {"conversations": []}

        with patch(
            "agents.toolmind_harness.agent.run_harness", side_effect=fake_run_harness
        ):
            exit_code = ToolmindAgent().run(
                instruction="test",
                cfg=runtime,
                console=Console(record=True),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured_kwargs["temperature"], 0.0)
        self.assertFalse(captured_kwargs["strict_protocol"])
        self.assertTrue(captured_kwargs["allow_fallback_search"])
        self.assertFalse(captured_kwargs["force_think_tag"])
        self.assertFalse(captured_kwargs["request_reasoning"])
        self.assertFalse(captured_kwargs["internal_protocol_retry"])
        self.assertTrue(captured_kwargs["record_protocol_repairs"])

    def test_run_normalizes_openai_prefixed_model_for_openai_apis(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="toolmind-harness",
            model=AgentModelConfig(
                model="openai/local-model",
                api_base="http://127.0.0.1:1234/v1",
                api_key="k",
            ),
            agent_config={},
        )
        captured_kwargs: dict[str, object] = {}

        def fake_run_harness(**kwargs):  # type: ignore[no-untyped-def]
            captured_kwargs.update(kwargs)
            return {"conversations": []}

        with patch(
            "agents.toolmind_harness.agent.run_harness", side_effect=fake_run_harness
        ):
            exit_code = ToolmindAgent().run(
                instruction="test",
                cfg=runtime,
                console=Console(record=True),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured_kwargs["model"], "local-model")


class TestRegistryAndParser(unittest.TestCase):
    def test_get_agent_unknown_lists_available_agents(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available agents: claude, qwen, terminus-2, toolmind-harness",
        ):
            get_agent("missing-agent")

    def test_mcp_parser_parse_all_parses_valid_and_fallback_arguments(self) -> None:
        text = (
            "<use_mcp_tool><server_name>tool-python</server_name>"
            "<tool_name>run_command</tool_name>"
            '<arguments>{"sandbox_id":"s1","command":"pwd"}</arguments></use_mcp_tool>'
            "<use_mcp_tool><server_name>tool-python</server_name>"
            "<tool_name>run_python_code</tool_name>"
            "<arguments>{invalid json</arguments></use_mcp_tool>"
        )
        calls = harness.MCPParser.parse_all(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].arguments["sandbox_id"], "s1")
        self.assertIn("_raw_arguments", calls[1].arguments)


class TestOpenAiCompat(unittest.TestCase):
    def test_detect_provider_kind_distinguishes_openrouter_and_local(self) -> None:
        self.assertEqual(
            detect_provider_kind(api_base="https://openrouter.ai/api/v1"),
            "openrouter",
        )
        self.assertEqual(
            detect_provider_kind(api_base="http://127.0.0.1:1234/v1"),
            "openai_compatible_local",
        )

    def test_preflight_keeps_qwen_model_checks_runtime_based(self) -> None:
        message = preflight_agent_model_compatibility(
            agent_key="qwen",
            model="gpt-5.3-codex",
            api_base="https://api.openai.com/v1",
        )
        self.assertIsNone(message)


class TestToolExecutor(unittest.TestCase):
    def test_google_search_requires_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = harness.ToolExecutor(scratch_dir=Path(temp_dir))
            result = executor.execute(
                harness.ToolCall(
                    server_name="search_and_scrape_webpage",
                    tool_name="google_search",
                    arguments={},
                )
            )
        self.assertFalse(result["success"])
        self.assertIn("Missing required argument 'q'", result["error"])

    def test_google_search_fallback_scrapes_ddg_when_enabled(self) -> None:
        html = '<a class="result__a" href="https://example.com">Example Result</a>'
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = harness.ToolExecutor(
                scratch_dir=Path(temp_dir),
                allow_fallback_search=True,
            )
            with patch("urllib.request.urlopen", return_value=_FakeResponse(html)):
                result = executor.execute(
                    harness.ToolCall(
                        server_name="search_and_scrape_webpage",
                        tool_name="google_search",
                        arguments={"q": "example"},
                    )
                )
        self.assertTrue(result["success"])
        self.assertEqual(result["results"][0]["title"], "Example Result")

    def test_scrape_and_extract_prefixes_target_without_extractor_model(self) -> None:
        body = "First line\n" + (
            "A useful sentence with enough characters to include.\n" * 2
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = harness.ToolExecutor(scratch_dir=Path(temp_dir))
            with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
                result = executor.execute(
                    harness.ToolCall(
                        server_name="jina_scrape_llm_summary",
                        tool_name="scrape_and_extract_info",
                        arguments={
                            "url": "https://example.com",
                            "info_to_extract": "founding year",
                        },
                    )
                )
        self.assertTrue(result["success"])
        self.assertIn(
            "Requested extraction target: founding year", result["extracted_info"]
        )


class TestHeadlessAgents(unittest.TestCase):
    def test_qwen_headless_builds_expected_env_and_args(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="qwen/qwen3.5-35b-a3b",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                temperature=0.0,
            ),
            agent_config={
                "binary": "qwen-custom",
                "token_limit": 4096,
                "sampling_params": {"temperature": 0.1},
                "mcp_servers": {
                    "tool-python": {
                        "command": "python3",
                        "args": ["mcp.py"],
                        "cwd": "/tmp",
                        "env": {},
                        "timeout": 30,
                    }
                },
                "env": {"CUSTOM_ENV": "1"},
            },
        )
        captured: dict[str, object] = {}

        def fake_run_subprocess(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 0

        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=fake_run_subprocess,
        ):
            code = QwenHeadlessAgent().run(
                instruction="do a quick check",
                cfg=runtime,
                console=Console(record=True),
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            captured["args"],
            ["qwen-custom", "-p", "do a quick check", "-y"],
        )
        env = cast(dict[str, str], captured["env"])
        self.assertEqual(env["OPENAI_MODEL"], "qwen/qwen3.5-35b-a3b")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env["OPENAI_API_BASE"], "https://openrouter.ai/api/v1")
        self.assertEqual(env["OPENAI_API_KEY"], "test-key")
        self.assertEqual(env["CUSTOM_ENV"], "1")
        self.assertTrue(env["QWEN_CODE_SYSTEM_SETTINGS_PATH"].endswith(".json"))
        self.assertTrue(env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"].endswith(".json"))
        self.assertNotEqual(env["HOME"], str(Path.cwd()))
        self.assertTrue(env["XDG_CONFIG_HOME"].startswith(env["HOME"]))

    def test_claude_builds_expected_env_and_args(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="claude",
            model=AgentModelConfig(
                model="claude-sonnet-4-6",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                temperature=0.0,
            ),
            agent_config={
                "binary": "claude-custom",
                "output_format": "json",
                "skip_permissions": True,
                "pass_model_arg": True,
                "isolate_home": True,
                "allowed_tools": ["Bash(ls:*)"],
                "env": {"CUSTOM_ENV": "1"},
            },
        )
        captured: dict[str, object] = {}

        def fake_run_subprocess(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 0

        with patch(
            "agents.claude.claude_agent.run_subprocess",
            side_effect=fake_run_subprocess,
        ):
            code = ClaudeCodeAgent().run(
                instruction="do a quick check",
                cfg=runtime,
                console=Console(record=True),
            )

        self.assertEqual(code, 0)
        args = cast(list[str], captured["args"])
        self.assertEqual(args[0], "claude-custom")
        self.assertIn("--model", args)
        model_idx = args.index("--model")
        self.assertEqual(args[model_idx + 1], "claude-sonnet-4-6")
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertIn("--allowedTools", args)
        env = cast(dict[str, str], captured["env"])
        self.assertEqual(env["OPENAI_MODEL"], "claude-sonnet-4-6")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env["OPENAI_API_BASE"], "https://openrouter.ai/api/v1")
        self.assertEqual(env["OPENAI_API_KEY"], "test-key")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "test-key")
        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-sonnet-4-6")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "test-key")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env["CUSTOM_ENV"], "1")
        self.assertNotEqual(env["HOME"], str(Path.cwd()))

    def test_claude_can_disable_model_arg_and_home_isolation(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="claude",
            model=AgentModelConfig(
                model="qwen/qwen3.5-35b-a3b",
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            agent_config={
                "binary": "claude-custom",
                "pass_model_arg": False,
                "isolate_home": False,
            },
        )
        captured: dict[str, object] = {}

        def fake_run_subprocess(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 0

        with patch(
            "agents.claude.claude_agent.run_subprocess",
            side_effect=fake_run_subprocess,
        ):
            code = ClaudeCodeAgent().run(
                instruction="do a quick check",
                cfg=runtime,
                console=Console(record=True),
            )
        self.assertEqual(code, 0)
        args = cast(list[str], captured["args"])
        self.assertNotIn("--model", args)
        env = cast(dict[str, str], captured["env"])
        self.assertEqual(env["ANTHROPIC_MODEL"], "qwen/qwen3.5-35b-a3b")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "test-key")
        self.assertNotIn("XDG_CONFIG_HOME", env)

    def test_headless_agents_normalize_openai_prefixed_model(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="openai/local-model",
                api_base="http://127.0.0.1:1234/v1",
                api_key="test-key",
            ),
        )
        qwen_captured: dict[str, object] = {}

        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=lambda **kwargs: qwen_captured.update(kwargs) or 0,
        ):
            self.assertEqual(
                QwenHeadlessAgent().run(
                    instruction="test",
                    cfg=runtime,
                    console=Console(record=True),
                ),
                0,
            )

        self.assertEqual(
            cast(dict[str, str], qwen_captured["env"])["OPENAI_MODEL"],
            "local-model",
        )

    def test_new_agent_error_paths_return_nonzero(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="claude",
            model=AgentModelConfig(
                model="claude-sonnet-4-6",
                api_base="https://example.invalid/v1",
                api_key="k",
            ),
        )
        console = Console(record=True)
        with patch(
            "agents.claude.claude_agent.run_subprocess",
            side_effect=FileNotFoundError(),
        ):
            code = ClaudeCodeAgent().run(
                instruction="test",
                cfg=runtime,
                console=console,
            )
        self.assertEqual(code, 1)
        self.assertIn("claude CLI not found", console.export_text())

    def test_qwen_reports_actionable_chat_compat_error_for_non_chat_model(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="qwen",
            model=AgentModelConfig(
                model="gpt-5.3-codex",
                api_base="https://api.openai.com/v1",
                api_key="k",
            ),
            agent_config={"binary": "qwen-custom"},
        )
        console = Console(record=True)
        with patch(
            "agents.qwen.qwen_agent.run_subprocess",
            side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd=["qwen-custom", "-p", "test", "-y"],
                stderr=(
                    "This is not a chat model and thus not supported in the v1/chat/"
                    "completions endpoint. Did you mean to use v1/completions?"
                ),
            ),
        ) as run_mock:
            code = QwenHeadlessAgent().run(
                instruction="test",
                cfg=runtime,
                console=console,
            )
        self.assertEqual(code, 1)
        run_mock.assert_called_once()
        output = console.export_text()
        self.assertIn("Qwen Code cannot use model 'gpt-5.3-codex'", output)
        self.assertIn("terminus-2", output)

    def test_claude_accepts_non_claude_model_for_openrouter_proxy(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="claude",
            model=AgentModelConfig(
                model="qwen/qwen3.5-35b-a3b",
                api_base="https://openrouter.ai/api/v1",
                api_key="k",
            ),
            agent_config={"binary": "claude-custom"},
        )
        captured: dict[str, object] = {}

        def fake_run_subprocess(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 0

        with patch(
            "agents.claude.claude_agent.run_subprocess",
            side_effect=fake_run_subprocess,
        ):
            code = ClaudeCodeAgent().run(
                instruction="test",
                cfg=runtime,
                console=Console(record=True),
            )
        self.assertEqual(code, 0)
        env = cast(dict[str, str], captured["env"])
        self.assertEqual(env["ANTHROPIC_MODEL"], "qwen/qwen3.5-35b-a3b")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "k")

    def test_claude_surfaces_actionable_message_on_subprocess_failure(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="claude",
            model=AgentModelConfig(
                model="gpt-5.3-codex",
                api_base="https://api.openai.com/v1",
                api_key="k",
            ),
            agent_config={
                "pass_model_arg": True,
                "isolate_home": False,
            },
        )
        console = Console(record=True)
        with patch(
            "agents.claude.claude_agent.run_subprocess",
            side_effect=subprocess.CalledProcessError(2, ["claude"]),
        ):
            code = ClaudeCodeAgent().run(
                instruction="test",
                cfg=runtime,
                console=console,
            )
        self.assertEqual(code, 1)
        output = console.export_text()
        self.assertIn("claude CLI exited with status 2", output)
        self.assertIn("agents.claude.pass_model_arg", output)
        self.assertIn("agents.claude.isolate_home", output)


class TestHarnessLoop(unittest.TestCase):
    def test_run_harness_returns_done_without_tool_when_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "traj.json"
            fake_client = _FakeClient(
                responses=[
                    harness.CompletionResult(
                        content="Final answer without tools.",
                        reasoning="",
                    )
                ]
            )
            with (
                patch(
                    "agents.toolmind_harness.harness.OpenAIChatClient",
                    return_value=fake_client,
                ),
                patch(
                    "agents.toolmind_harness.harness.ToolExecutor",
                    return_value=_FakeTools(),
                ),
            ):
                row = harness.run_harness(
                    question="q",
                    model="m",
                    output_path=output_path,
                    key="k",
                    row_id="id",
                    max_assistant_turns=3,
                    temperature=0.1,
                    strict_protocol=False,
                    min_tool_turns=1,
                    repair_attempts=0,
                    allow_fallback_search=False,
                    force_think_tag=False,
                    request_reasoning=False,
                    internal_protocol_retry=False,
                    max_internal_protocol_retries=0,
                    record_protocol_repairs=False,
                    api_key="x",
                    api_base="https://example.invalid/v1",
                )
        self.assertEqual(row["conversations"][-1]["role"], "assistant")
        self.assertIn(
            "Final answer without tools.", row["conversations"][-1]["content"]
        )

    def test_run_harness_records_missing_tool_protocol_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "traj.json"
            fake_client = _FakeClient(
                responses=[
                    harness.CompletionResult(
                        content="No tool call here.",
                        reasoning="",
                    )
                ]
            )
            issues: list[tuple[str, str]] = []
            callbacks = harness.HarnessCallbacks(
                on_issue=lambda kind, msg: issues.append((kind, msg))
            )
            with (
                patch(
                    "agents.toolmind_harness.harness.OpenAIChatClient",
                    return_value=fake_client,
                ),
                patch(
                    "agents.toolmind_harness.harness.ToolExecutor",
                    return_value=_FakeTools(),
                ),
            ):
                row = harness.run_harness(
                    question="q",
                    model="m",
                    output_path=output_path,
                    key="k",
                    row_id="id",
                    max_assistant_turns=1,
                    temperature=0.1,
                    strict_protocol=True,
                    min_tool_turns=1,
                    repair_attempts=0,
                    allow_fallback_search=False,
                    force_think_tag=False,
                    request_reasoning=False,
                    internal_protocol_retry=False,
                    max_internal_protocol_retries=0,
                    record_protocol_repairs=True,
                    api_key="x",
                    api_base="https://example.invalid/v1",
                    callbacks=callbacks,
                )
        self.assertEqual(issues[0][0], "protocol")
        self.assertEqual(
            row["conversations"][-1]["content"], harness.REPAIR_MSG_MISSING_TOOL
        )


class TestTerminus2Agent(unittest.TestCase):
    def test_run_builds_core_config_and_returns_run_agent_code(self) -> None:
        runtime = AgentRuntimeConfig(
            agent_key="terminus-2",
            model=AgentModelConfig(
                model="model-y",
                api_base="https://example.invalid/v1",
                api_key="api",
                temperature=0.5,
            ),
            agent_config={
                "verbosity": 0,
                "max_turns": 7,
                "max_wait_seconds": 3.5,
                "final_message": False,
            },
        )
        captured: dict[str, object] = {}

        def fake_run_agent(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return 9

        with patch("agents.terminus2.agent.run_agent", side_effect=fake_run_agent):
            exit_code = terminus_agent.Terminus2Agent().run(
                instruction="inspect",
                cfg=runtime,
                console=Console(record=True),
            )

        self.assertEqual(exit_code, 9)
        cfg = cast(terminus_agent.CoreConfig, captured["cfg"])
        self.assertEqual(cfg.max_turns, 7)
        self.assertEqual(cfg.max_wait_seconds, 3.5)
        self.assertFalse(cfg.final_message_enabled)
        self.assertEqual(captured["instruction"], "inspect")
        self.assertEqual(captured["api_key"], "api")

    def test_render_issue_output_hides_non_model_issues_at_verbosity_zero(self) -> None:
        console = Console(record=True, width=60)
        terminus_agent._render_issue_output(
            console=console,
            kind="parser",
            message="bad json",
            verbosity=0,
        )
        self.assertEqual(console.export_text(), "")

        terminus_agent._render_issue_output(
            console=console,
            kind="model",
            message="rate limited",
            verbosity=0,
        )
        self.assertIn("error: model", console.export_text())


if __name__ == "__main__":
    unittest.main()
