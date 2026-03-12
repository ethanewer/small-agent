import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "agents" / "terminus2" / "core_agent.py"
CLI_PATH = PROJECT_ROOT / "cli.py"
PROMPT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "terminus-json-plain.txt"

core_spec = importlib.util.spec_from_file_location("core_agent", CORE_PATH)
assert core_spec and core_spec.loader
core_agent = importlib.util.module_from_spec(core_spec)
sys.modules["core_agent"] = core_agent
core_spec.loader.exec_module(core_agent)

summary_spec = importlib.util.spec_from_file_location(
    "final_summary", PROJECT_ROOT / "agents" / "terminus2" / "final_summary.py"
)
assert summary_spec and summary_spec.loader
final_summary = importlib.util.module_from_spec(summary_spec)
sys.modules["final_summary"] = final_summary
summary_spec.loader.exec_module(final_summary)

cli_spec = importlib.util.spec_from_file_location("cli", CLI_PATH)
assert cli_spec and cli_spec.loader
cli = importlib.util.module_from_spec(cli_spec)
sys.modules["cli"] = cli
cli_spec.loader.exec_module(cli)


class FakeSession:
    def __init__(self) -> None:
        self.send_keys_calls: list[list[str]] = []
        self.closed = False
        self._incremental_output = "Current Terminal Screen:\n(empty)"

    def send_keys(
        self,
        keys: str | list[str],
        *,
        min_timeout_sec: float = 0.0,
    ) -> None:
        del min_timeout_sec
        if isinstance(keys, str):
            keys = [keys]
        self.send_keys_calls.append(keys)

    def get_incremental_output(self) -> str:
        return self._incremental_output

    def is_session_alive(self) -> bool:
        return not self.closed

    def capture_pane(self, *, capture_entire: bool = False) -> str:
        del capture_entire
        return ""

    def close(self) -> None:
        self.closed = True


def _as_any(value: object) -> Any:
    return value


class TestPromptParity(unittest.TestCase):
    def test_system_prompt_matches_terminus2_template(self) -> None:
        expected = PROMPT_FIXTURE_PATH.read_text()
        self.assertEqual(core_agent.SYSTEM_PROMPT, expected)

    def test_build_prompt_matches_rendered_reference_template(self) -> None:
        expected_template = PROMPT_FIXTURE_PATH.read_text()
        instruction = "List files"
        terminal_state = "Current Terminal Screen:\n(empty)"
        expected = expected_template.format(
            instruction=instruction, terminal_state=terminal_state
        )
        actual = core_agent.build_prompt(
            instruction=instruction,
            terminal_state=terminal_state,
            max_wait_seconds=60.0,
        )
        self.assertEqual(actual, expected)

    def test_completion_confirmation_message_matches_terminus2(self) -> None:
        output = "line one\nline two"
        expected = (
            f"Current terminal state:\n{output}\n\n"
            "Are you sure you want to mark the task as complete? "
            "This will trigger your solution to be graded and you won't be able to "
            'make any further corrections. If so, include "task_complete": true '
            "in your JSON response again."
        )
        self.assertEqual(core_agent.completion_confirmation_message(output), expected)


class TestParserParity(unittest.TestCase):
    def test_extract_json_content_handles_wrapped_text(self) -> None:
        response = 'prefix text {"a": 1, "nested": {"b": 2}} suffix text'
        content, warnings = core_agent._extract_json_content(response)
        self.assertEqual(content, '{"a": 1, "nested": {"b": 2}}')
        self.assertTrue(any("before" in w for w in warnings))
        self.assertTrue(any("after" in w for w in warnings))

    def test_extract_json_content_handles_escaped_quote_and_brace(self) -> None:
        response = 'noise {"text": "quote \\" and brace }", "ok": true} trailing'
        content, _ = core_agent._extract_json_content(response)
        self.assertEqual(
            content,
            '{"text": "quote \\" and brace }", "ok": true}',
        )

    def test_parse_response_coerces_task_complete_string(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n"}],
          "task_complete": "yes"
        }
        """
        result = core_agent.parse_response(text)
        self.assertEqual(result.error, "")
        assert result.parsed is not None
        self.assertTrue(result.parsed.task_complete)
        self.assertEqual(result.parsed.commands[0].duration, 1.0)

    def test_parse_response_reports_missing_fields_in_single_error(self) -> None:
        text = """
        {
          "analysis": "a",
          "commands": []
        }
        """
        result = core_agent.parse_response(text)
        self.assertIn("Missing required fields: plan", result.error)

    def test_parse_response_invalid_duration_falls_back_to_default(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n", "duration": "slow"}]
        }
        """
        result = core_agent.parse_response(text)
        self.assertEqual(result.error, "")
        assert result.parsed is not None
        self.assertEqual(result.parsed.commands[0].duration, 1.0)

    def test_parse_response_tolerates_bad_commands_when_task_complete(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"duration": 1.0}],
          "task_complete": true
        }
        """
        result = core_agent.parse_response(text)
        self.assertEqual(result.error, "")
        assert result.parsed is not None
        self.assertTrue(result.parsed.task_complete)
        self.assertEqual(result.parsed.commands, [])

    def test_parse_response_rejects_invalid_final_message_type(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n"}],
          "final_message": 42
        }
        """
        result = core_agent.parse_response(text)
        self.assertNotEqual(result.error, "")

    def test_parse_response_auto_fixes_incomplete_json(self) -> None:
        text = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n","duration":0.1}]'
        result = core_agent.parse_response(text)
        self.assertEqual(result.error, "")
        assert result.parsed is not None
        self.assertIn("AUTO-CORRECTED", result.warning)

    def test_parse_response_warns_on_extra_text(self) -> None:
        text = 'Here is my response: {"analysis":"a","plan":"p","commands":[]}'
        result = core_agent.parse_response(text)
        self.assertEqual(result.error, "")
        self.assertIn("Extra text", result.warning)


class TestExecutionAndLoop(unittest.TestCase):
    def test_limit_output_length_short_output_unchanged(self) -> None:
        text = "hello"
        self.assertEqual(core_agent.limit_output_length(text, max_bytes=20), text)

    def test_limit_output_length_long_output_contains_marker(self) -> None:
        text = "abcdef" * 200
        limited = core_agent.limit_output_length(text, max_bytes=40)
        self.assertIn("output limited to 40 bytes", limited)
        self.assertTrue(limited.startswith("abcdef"))
        self.assertTrue(limited.endswith("abcdef"))

    def test_execute_command_wait_keystrokes_only_sleeps(self) -> None:
        session = FakeSession()
        cmd = core_agent.Command(keystrokes="", duration=0.25)
        with patch("time.sleep") as sleep_mock:
            core_agent.execute_command(
                session=_as_any(session),
                cmd=cmd,
                max_wait_seconds=60.0,
            )
        sleep_mock.assert_called_once_with(0.25)
        self.assertEqual(session.send_keys_calls, [])

    def test_execute_command_single_newline_uses_send_keys(self) -> None:
        session = FakeSession()
        cmd = core_agent.Command(keystrokes="echo hi\n", duration=0.1)
        core_agent.execute_command(
            session=_as_any(session),
            cmd=cmd,
            max_wait_seconds=60.0,
        )
        self.assertEqual(session.send_keys_calls, [["echo hi\n"]])

    def test_run_agent_uses_terminus2_parse_error_feedback_and_confirmation(
        self,
    ) -> None:
        cfg = core_agent.Config(
            active_model_key="test",
            active_model=core_agent.ModelConfig(model="x", api_base="y"),
            max_turns=6,
        )
        session = FakeSession()
        prompts: list[str] = []
        responses = iter(
            [
                "not json",
                '{"analysis":"a","plan":"p","commands":[],"task_complete":true}',
                '{"analysis":"a2","plan":"p2","commands":[],"task_complete":true}',
                "post-run summary",
            ]
        )

        def fake_call_model(
            cfg: Any,
            prompt: str,
            history: list[dict[str, str]],
            api_key: str,
        ) -> Any:
            del cfg, history, api_key
            prompts.append(prompt)
            return final_summary.ModelResult(
                content=next(responses),
                prompt_tokens=0,
                completion_tokens=0,
            )

        with (
            patch.object(core_agent, "start_session", return_value=session),
            patch.object(core_agent, "call_model", side_effect=fake_call_model),
        ):
            exit_code = core_agent.run_agent(
                instruction="do thing",
                cfg=cfg,
                api_key="k",
            )

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(prompts), 4)
        self.assertIn(
            "Please fix these issues and provide a proper JSON response.", prompts[1]
        )
        incremental = session.get_incremental_output()
        self.assertEqual(
            prompts[2], core_agent.completion_confirmation_message(incremental)
        )
        self.assertEqual(prompts[3], final_summary.post_run_summary_prompt())
        self.assertTrue(session.closed)

    def test_run_agent_skips_final_summary_prompt_when_disabled(self) -> None:
        cfg = core_agent.Config(
            active_model_key="test",
            active_model=core_agent.ModelConfig(model="x", api_base="y"),
            max_turns=4,
            final_message_enabled=False,
        )
        session = FakeSession()
        prompts: list[str] = []
        done_messages: list[str] = []
        responses = iter(
            [
                '{"analysis":"a","plan":"p","commands":[],"task_complete":true}',
                '{"analysis":"a2","plan":"p2","commands":[],"task_complete":true}',
            ]
        )

        def fake_call_model(
            cfg: Any,
            prompt: str,
            history: list[dict[str, str]],
            api_key: str,
        ) -> Any:
            del cfg, history, api_key
            prompts.append(prompt)
            return final_summary.ModelResult(
                content=next(responses),
                prompt_tokens=0,
                completion_tokens=0,
            )

        callbacks = core_agent.AgentCallbacks(
            on_done=lambda done_text: done_messages.append(done_text)
        )
        with (
            patch.object(core_agent, "start_session", return_value=session),
            patch.object(core_agent, "call_model", side_effect=fake_call_model),
        ):
            exit_code = core_agent.run_agent(
                instruction="do thing",
                cfg=cfg,
                api_key="k",
                callbacks=callbacks,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(prompts), 2)
        incremental = session.get_incremental_output()
        self.assertEqual(
            prompts[1], core_agent.completion_confirmation_message(incremental)
        )
        self.assertEqual(done_messages, [])
        self.assertTrue(session.closed)


class TestTlsHandling(unittest.TestCase):
    def test_is_tls_certificate_error_detects_common_marker(self) -> None:
        self.assertTrue(
            core_agent._is_tls_certificate_error(
                message=(
                    "APIError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
                    "failed: self-signed certificate in certificate chain"
                )
            )
        )

    def test_call_model_tls_error_raises_actionable_message(self) -> None:
        cfg = core_agent.Config(
            active_model_key="test",
            active_model=core_agent.ModelConfig(
                model="qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
            ),
        )
        tls_error = Exception(
            "OpenrouterException - [SSL: CERTIFICATE_VERIFY_FAILED] "
            "certificate verify failed: self-signed certificate in certificate chain"
        )

        class _FakeCompletions:
            def create(self, **_kwargs: Any) -> None:
                raise tls_error

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        with patch.object(
            core_agent,
            "_make_openai_client",
            return_value=_FakeClient(),
        ):
            with self.assertRaisesRegex(RuntimeError, "SMALL_AGENT_CA_BUNDLE"):
                core_agent.call_model(
                    cfg=cfg,
                    prompt="ping",
                    history=[],
                    api_key="dummy-key",
                )


class TestFinalSummaryNormalization(unittest.TestCase):
    def test_normalize_summary_response_prefers_final_message_from_json(self) -> None:
        normalized = final_summary.normalize_summary_response(
            '{"analysis":"a","plan":"p","commands":[],"final_message":"Done cleanly."}'
        )
        self.assertEqual(normalized, "Done cleanly.")

    def test_normalize_summary_response_strips_code_fences(self) -> None:
        normalized = final_summary.normalize_summary_response(
            "```markdown\nCompleted successfully.\n```"
        )
        self.assertEqual(normalized, "Completed successfully.")


class TestResolveApiKey(unittest.TestCase):
    def test_resolve_api_key_prefers_literal_config_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("literal-key"), "literal-key")

    def test_resolve_api_key_uses_dollar_variable(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "expanded-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("$OPENAI_API_KEY"), "expanded-key")

    def test_resolve_api_key_uses_variable_name_without_dollar(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "named-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("OPENAI_API_KEY"), "named-key")

    def test_resolve_api_key_falls_back_to_subprocess_for_env_var(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="from-zsh",
                    stderr="",
                )
                self.assertEqual(cli.resolve_api_key("OPENAI_API_KEY"), "from-zsh")


class TestModelSelection(unittest.TestCase):
    def test_resolve_model_key_uses_cli_model_when_provided(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                ),
                "b": cli.ConfigModelEntry(
                    model="model-b",
                    api_base="https://example.com/v1",
                    api_key="KEY_B",
                    temperature=0.7,
                ),
            },
            default_agent="terminus-2",
            agents={"terminus-2": {}},
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )
        self.assertEqual(
            cli.resolve_model_key(
                config=loaded, cli_model_key="b", selected_model_key=None
            ),
            "b",
        )

    def test_resolve_model_key_raises_for_unknown_cli_model(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                )
            },
            default_agent="terminus-2",
            agents={"terminus-2": {}},
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )
        with self.assertRaisesRegex(ValueError, "Unknown model key"):
            cli.resolve_model_key(
                config=loaded, cli_model_key="missing", selected_model_key=None
            )


class TestInteractiveCommands(unittest.TestCase):
    def _loaded_config(self) -> Any:
        return cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                )
            },
            default_agent="terminus-2",
            agents={"terminus-2": {}, "qwen": {}},
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )

    def test_parse_interactive_command_sets_verbosity_inline(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/verbosity 1",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.updated_verbosity, 1)
        self.assertEqual(result.instruction, "")

    def test_parse_interactive_command_prompts_for_missing_verbosity(self) -> None:
        loaded = self._loaded_config()
        with patch.object(cli.Prompt, "ask", return_value="1"):
            result = cli.parse_interactive_command(
                console=cli.Console(record=True),
                instruction="/verbosity",
                config=loaded,
            )
        self.assertTrue(result.handled)
        self.assertEqual(result.updated_verbosity, 1)

    def test_parse_interactive_command_sets_max_turns(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/max_turns 12",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(loaded.max_turns, 12)
        self.assertIsNone(result.updated_verbosity)

    def test_parse_interactive_command_sets_max_wait_seconds(self) -> None:
        loaded = self._loaded_config()
        with patch.object(cli.Prompt, "ask", return_value="2.5"):
            result = cli.parse_interactive_command(
                console=cli.Console(record=True),
                instruction="/max_wait_seconds",
                config=loaded,
            )
        self.assertTrue(result.handled)
        self.assertAlmostEqual(loaded.max_wait_seconds, 2.5)

    def test_parse_interactive_command_unknown_slash_command_is_handled(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/unknown",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.instruction, "")

    def test_parse_interactive_command_empty_input_shows_help_and_is_handled(
        self,
    ) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.instruction, "")

    def test_parse_interactive_command_sets_agent(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/agent qwen",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.selected_agent, "qwen")

    def test_parse_interactive_command_prompts_for_missing_agent(self) -> None:
        loaded = self._loaded_config()
        with patch.object(cli.Prompt, "ask", return_value="2"):
            result = cli.parse_interactive_command(
                console=cli.Console(record=True),
                instruction="/agent",
                config=loaded,
            )
        self.assertTrue(result.handled)
        self.assertEqual(result.selected_agent, "qwen")


class TestPlanModeFlag(unittest.TestCase):
    def test_parse_args_accepts_plan_flag(self) -> None:
        args = cli.parse_args(["--plan", "hello", "world"])
        self.assertTrue(args.plan)
        self.assertEqual(args.instruction, ["hello", "world"])

    def test_main_wires_plan_mode_into_runtime_config(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="openai/gpt-4o-mini",
                    api_base="https://api.openai.com/v1",
                    api_key="literal-key",
                    temperature=0.0,
                )
            },
            default_agent="liteforge",
            agents={"liteforge": {}},
            verbosity=0,
            max_turns=5,
            max_wait_seconds=10.0,
        )
        captured_cfg: dict[str, Any] = {}

        class _Result:
            exit_code = 0

        def fake_runner(*, agent, task, cfg, console, sink):  # type: ignore[no-untyped-def]
            del agent, task, console, sink
            captured_cfg["cfg"] = cfg
            return _Result()

        with (
            patch.object(cli, "load_config", return_value=loaded),
            patch.object(cli, "available_agents", return_value={"liteforge": object()}),
            patch.object(cli, "get_agent", return_value=object()),
            patch.object(cli, "run_agent_task_with_fallback", side_effect=fake_runner),
            patch.object(
                sys,
                "argv",
                [
                    "cli.py",
                    "--agent",
                    "liteforge",
                    "--model",
                    "a",
                    "--plan",
                    "draft a plan",
                ],
            ),
        ):
            with self.assertRaises(SystemExit) as ex:
                cli.main()

        self.assertEqual(ex.exception.code, 0)
        cfg = captured_cfg["cfg"]
        self.assertEqual(cfg.agent_config.get("plan_mode"), True)
        self.assertEqual(cfg.agent_config.get("readonly"), True)


class TestAgentSelection(unittest.TestCase):
    def _loaded_config(self) -> Any:
        return cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                )
            },
            default_agent="terminus-2",
            agents={"terminus-2": {}, "qwen": {}},
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )

    def test_resolve_agent_key_uses_cli_override(self) -> None:
        loaded = self._loaded_config()
        self.assertEqual(
            cli.resolve_agent_key(
                config=loaded,
                cli_agent_key="qwen",
                selected_agent_key=None,
            ),
            "qwen",
        )

    def test_resolve_agent_key_uses_selected_when_no_cli(self) -> None:
        loaded = self._loaded_config()
        self.assertEqual(
            cli.resolve_agent_key(
                config=loaded,
                cli_agent_key=None,
                selected_agent_key="qwen",
            ),
            "qwen",
        )


class TestLoadConfigValidation(unittest.TestCase):
    def _write_config(self, payload: dict[str, object]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with temp_file:
            json.dump(payload, temp_file)
        return Path(temp_file.name)

    def test_load_config_defaults_agents_when_missing(self) -> None:
        path = self._write_config(
            {
                "default_model": "qwen3-coder-next",
                "models": {
                    "qwen3-coder-next": {
                        "model": "qwen/qwen3-coder-next",
                        "api_base": "https://openrouter.ai/api/v1",
                        "api_key": "OPENROUTER_API_KEY",
                    }
                },
            }
        )
        try:
            loaded = cli.load_config(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.default_agent, "terminus-2")
        self.assertEqual(loaded.agents, {"terminus-2": {}})

    def test_load_config_maps_legacy_verbosity_3_to_1(self) -> None:
        path = self._write_config(
            {
                "default_model": "qwen3-coder-next",
                "models": {
                    "qwen3-coder-next": {
                        "model": "qwen/qwen3-coder-next",
                        "api_base": "https://openrouter.ai/api/v1",
                        "api_key": "OPENROUTER_API_KEY",
                    }
                },
                "default_agent": "terminus-2",
                "agents": {"terminus-2": {}},
                "verbosity": 3,
            }
        )
        try:
            loaded = cli.load_config(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.verbosity, 1)

    def test_load_config_rejects_unknown_default_model(self) -> None:
        path = self._write_config(
            {
                "default_model": "missing",
                "models": {
                    "qwen3-coder-next": {
                        "model": "qwen/qwen3-coder-next",
                        "api_base": "https://openrouter.ai/api/v1",
                        "api_key": "OPENROUTER_API_KEY",
                    }
                },
                "default_agent": "terminus-2",
                "agents": {"terminus-2": {}},
            }
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                "default_model must match a key in models",
            ):
                cli.load_config(path)
        finally:
            path.unlink(missing_ok=True)

    def test_load_config_rejects_default_agent_not_in_agents(self) -> None:
        path = self._write_config(
            {
                "default_model": "qwen3-coder-next",
                "models": {
                    "qwen3-coder-next": {
                        "model": "qwen/qwen3-coder-next",
                        "api_base": "https://openrouter.ai/api/v1",
                        "api_key": "OPENROUTER_API_KEY",
                    }
                },
                "default_agent": "qwen",
                "agents": {"terminus-2": {}},
            }
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                "default_agent must match a key in agents",
            ):
                cli.load_config(path)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_agent_key_error_is_actionable_for_removed_agents(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="qwen3-coder-next",
            models={
                "qwen3-coder-next": cli.ConfigModelEntry(
                    model="qwen/qwen3-coder-next",
                    api_base="https://openrouter.ai/api/v1",
                    api_key="OPENROUTER_API_KEY",
                    temperature=None,
                )
            },
            default_agent="terminus-2",
            agents={"terminus-2": {}, "qwen": {}},
            verbosity=0,
            max_turns=50,
            max_wait_seconds=60.0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "Unknown agent key 'claude'. Available agent keys: terminus-2, qwen",
        ):
            cli.resolve_agent_key(
                config=loaded,
                cli_agent_key="claude",
                selected_agent_key=None,
            )


if __name__ == "__main__":
    unittest.main()
