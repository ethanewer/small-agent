from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

from rich.console import Console

from agents.core.task import Task
from agents.interface import AgentModelConfig, AgentRuntimeConfig
from agents.liteforge.context import Context, ToolCall, ToolResult
from agents.liteforge.orchestrator import Orchestrator
from agents.liteforge.runtime_agent import LiteforgeAgent
from agents.liteforge.tools.executor import ToolExecutor
from agents.liteforge.tools.registry import ALL_TOOL_NAMES, READONLY_TOOL_NAMES
from agents.registry import available_agents


def _runtime_cfg(
    *,
    cwd: str | None = None,
    context_length: int | None = None,
    max_tokens: int | None = None,
) -> AgentRuntimeConfig:
    agent_config: dict[str, object] = {
        "stream": False,
        "max_requests_per_turn": 4,
        "max_tool_failure_per_turn": 2,
    }
    if cwd:
        agent_config["cwd"] = cwd

    if max_tokens is not None:
        agent_config["max_tokens"] = max_tokens

    return AgentRuntimeConfig(
        agent_key="liteforge",
        model=AgentModelConfig(
            model="openai/gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            context_length=context_length,
        ),
        agent_config=agent_config,
    )


def test_registry_includes_liteforge_agent() -> None:
    agents = available_agents()
    assert "liteforge" in agents
    assert isinstance(agents["liteforge"], LiteforgeAgent)


def test_liteforge_run_task_returns_success_and_final_message(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            recorded["executor_env"] = env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor, tools
            recorded["model"] = model
            recorded["max_requests_per_turn"] = max_requests_per_turn
            recorded["max_tool_failure_per_turn"] = max_tool_failure_per_turn
            recorded["stream"] = stream
            recorded["context"] = context

        def run(self) -> bool:
            recorded["openai_model"] = os.environ.get("OPENAI_MODEL")
            recorded["openai_base_url"] = os.environ.get("OPENAI_BASE_URL")
            context = recorded["context"]
            assert isinstance(context, Context)
            context.add_assistant_message(content="liteforge complete", tool_calls=None)
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    cfg = _runtime_cfg(cwd=str(tmp_path))
    console = Console(record=True)
    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="say hello", task_id="lf-1"),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    assert result.exit_code == 0
    assert result.task_id == "lf-1"
    assert result.final_message == "liteforge complete"
    assert recorded["model"] == "openai/gpt-4o-mini"
    assert recorded["max_requests_per_turn"] == 4
    assert recorded["max_tool_failure_per_turn"] == 2
    assert recorded["stream"] is False
    assert recorded["openai_model"] == "openai/gpt-4o-mini"
    assert recorded["openai_base_url"] == "https://api.openai.com/v1"

    executor_env = recorded["executor_env"]
    assert isinstance(executor_env, dict)
    assert executor_env["cwd"] == str(tmp_path)
    rendered = console.export_text()
    assert "Final Output" in rendered
    assert "liteforge complete" in rendered


def test_liteforge_run_task_maps_failed_orchestrator_to_nonzero(monkeypatch) -> None:
    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del context
            del executor
            del model
            del tools
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream

        def run(self) -> bool:
            return False

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="fail", task_id="lf-2"),
        cfg=_runtime_cfg(),
        console=Console(record=True),
        sink=None,
    )
    assert not result.success
    assert result.exit_code == 1


def test_liteforge_honors_disabled_final_output_panel(monkeypatch) -> None:
    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del tools
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self._context = context

        def run(self) -> bool:
            self._context.add_assistant_message(
                content="hidden final panel", tool_calls=None
            )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    cfg = _runtime_cfg()
    cfg.agent_config["final_message"] = False
    console = Console(record=True)
    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="hide final output", task_id="lf-final-0"
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    assert result.final_message == "hidden final panel"
    rendered = console.export_text()
    assert "Final Output" not in rendered


def test_liteforge_defaults_max_tokens_for_large_context(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del tools
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            recorded["max_tokens"] = context.max_tokens
            context.add_assistant_message(content="ok", tool_calls=None)

        def run(self) -> bool:
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="max tokens", task_id="lf-3"),
        cfg=_runtime_cfg(context_length=1_000_000),
        console=Console(record=True),
        sink=None,
    )
    assert result.success
    assert recorded["max_tokens"] == 20480


def test_liteforge_caps_configured_max_tokens_to_context_length(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del tools
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            recorded["max_tokens"] = context.max_tokens
            context.add_assistant_message(content="ok", tool_calls=None)

        def run(self) -> bool:
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="max tokens", task_id="lf-4"),
        cfg=_runtime_cfg(context_length=8000, max_tokens=50_000),
        console=Console(record=True),
        sink=None,
    )
    assert result.success
    assert recorded["max_tokens"] == 8000


def test_liteforge_plan_mode_replans_then_executes_with_shared_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, list[object]] = {
        "tool_name_sequences": [],
        "context_ids": [],
        "executor_ids": [],
        "contexts": [],
    }
    responses = iter(["Add a rollback step", "yes"])
    prompts: list[str] = []

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            tool_names = [tool["function"]["name"] for tool in tools]
            recorded["tool_name_sequences"].append(tool_names)
            recorded["context_ids"].append(id(context))
            recorded["executor_ids"].append(id(executor))
            recorded["contexts"].append(context)
            self.context = context
            self._context = context
            self._tool_names = tool_names

        def run(self) -> bool:
            if self._tool_names == list(READONLY_TOOL_NAMES):
                self._context.add_assistant_message(
                    content="Plan drafted", tool_calls=None
                )
            else:
                self._context.add_assistant_message(
                    content="Plan executed fully",
                    tool_calls=None,
                )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("builtins.input", fake_input)

    cfg = _runtime_cfg(cwd=str(tmp_path))
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="implement feature", task_id="lf-plan-1"
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    assert result.exit_code == 0
    assert result.final_message == "Plan executed fully"
    assert recorded["tool_name_sequences"] == [
        list(READONLY_TOOL_NAMES),
        list(READONLY_TOOL_NAMES),
        list(ALL_TOOL_NAMES),
    ]
    assert len(set(recorded["context_ids"])) == 1
    assert len(set(recorded["executor_ids"])) == 1

    final_context = recorded["contexts"][-1]
    assert isinstance(final_context, Context)
    user_texts = [
        msg.content or "" for msg in final_context.messages if msg.role == "user"
    ]
    assert any(
        "<feedback>Add a rollback step</feedback>" in text for text in user_texts
    )
    assert any(
        "The user has approved the plan. Now execute it completely." in text
        for text in user_texts
    )
    rendered = console.export_text()
    assert "Creating plan" in rendered
    assert "Executing plan" in rendered
    assert "PLAN MODE:" not in rendered
    assert prompts == [
        "Execute this plan? [y]es / [n]o / [feedback]: ",
        "Execute this plan? [y]es / [n]o / [feedback]: ",
    ]


def test_liteforge_plan_mode_rejects_without_execution_phase(monkeypatch) -> None:
    recorded_tool_name_sequences: list[list[str]] = []
    responses = iter(["no"])
    prompts: list[str] = []
    rendered_at_prompt: list[str] = []

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            tool_names = [tool["function"]["name"] for tool in tools]
            recorded_tool_name_sequences.append(tool_names)
            self.context = context
            self._context = context

        def run(self) -> bool:
            self._context.add_assistant_message(content="Plan drafted", tool_calls=None)
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    console = Console(record=True)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        rendered_at_prompt.append(console.export_text(clear=False))
        return next(responses)

    monkeypatch.setattr("builtins.input", fake_input)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="plan only", task_id="lf-plan-2"),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    assert result.exit_code == 0
    assert result.final_message == "Plan drafted"
    assert recorded_tool_name_sequences == [list(READONLY_TOOL_NAMES)]
    rendered = console.export_text()
    assert "Creating plan" in rendered
    assert "Executing plan" not in rendered
    assert "Plan rejected. Exiting." in rendered
    assert "Plan drafted" in rendered
    assert "Plan" in rendered
    assert "PLAN MODE:" not in rendered
    assert "─" in rendered
    assert "=" * 60 not in rendered
    assert prompts == ["Execute this plan? [y]es / [n]o / [feedback]: "]
    assert "Plan drafted" in rendered_at_prompt[0]


def test_orchestrator_tool_logs_use_compact_styled_lines() -> None:
    class FakeExecutor(ToolExecutor):
        def __init__(self) -> None:
            super().__init__(env={"cwd": "."})

        def execute(
            self, tool_name: str, arguments: dict[str, object]
        ) -> tuple[str, bool]:
            del tool_name, arguments
            return "ok", False

    orch = Orchestrator(
        context=Context(),
        executor=FakeExecutor(),
        model="model-x",
        tools=[],
        stream=True,
    )
    console = Console(record=True, width=70, stderr=True)
    orch.set_log_console(console=console)
    _ = orch._execute_tool_calls(
        [
            ToolCall(
                id="tool-1",
                name="shell",
                arguments={"description": "List files", "command": "ls"},
            ),
            ToolCall(
                id="tool-2",
                name="read",
                arguments={"file_path": "/tmp/demo.py"},
            ),
        ]
    )

    rendered = console.export_text()
    assert "─" in rendered
    assert "[shell] List files" in rendered
    assert "[read] /tmp/demo.py" in rendered
    assert "\n\n\n" not in rendered


def test_orchestrator_collapses_trailing_stream_newlines_before_tool_logs() -> None:
    class FakeExecutor(ToolExecutor):
        def __init__(self) -> None:
            super().__init__(env={"cwd": "."})

        def execute(
            self, tool_name: str, arguments: dict[str, object]
        ) -> tuple[str, bool]:
            del tool_name, arguments
            return "ok", False

    orch = Orchestrator(
        context=Context(),
        executor=FakeExecutor(),
        model="model-x",
        tools=[],
        stream=True,
    )
    console = Console(record=True, width=70, stderr=True)
    orch.set_log_console(console=console)
    stream_output = io.StringIO()
    with redirect_stdout(stream_output):
        orch._stream_callback("I'll explore this codebase.\n\n")
        _ = orch._execute_tool_calls(
            [
                ToolCall(
                    id="tool-1",
                    name="shell",
                    arguments={"description": "List files", "command": "ls"},
                )
            ]
        )

    assert stream_output.getvalue() == "I'll explore this codebase.\n"
    rendered = console.export_text()
    assert "[shell] List files" in rendered


def test_orchestrator_prints_separator_before_resumed_stream_text() -> None:
    class FakeExecutor(ToolExecutor):
        def __init__(self) -> None:
            super().__init__(env={"cwd": "."})

        def execute(
            self, tool_name: str, arguments: dict[str, object]
        ) -> tuple[str, bool]:
            del tool_name, arguments
            return "ok", False

    orch = Orchestrator(
        context=Context(),
        executor=FakeExecutor(),
        model="model-x",
        tools=[],
        stream=True,
    )
    console = Console(record=True, width=70, stderr=True)
    orch.set_log_console(console=console)
    stream_output = io.StringIO()
    with redirect_stdout(stream_output):
        _ = orch._execute_tool_calls(
            [
                ToolCall(
                    id="tool-1",
                    name="shell",
                    arguments={
                        "description": "List tests directory",
                        "command": "ls tests",
                    },
                )
            ]
        )
        orch._stream_callback("Based on my exploration...")

    rendered = console.export_text()
    assert "[shell] List tests directory" in rendered
    assert rendered.count("─") >= 2
    assert stream_output.getvalue() == "Based on my exploration..."


def test_orchestrator_prints_separator_when_requested_before_stream_text() -> None:
    orch = Orchestrator(
        context=Context(),
        executor=ToolExecutor(env={"cwd": "."}),
        model="model-x",
        tools=[],
        stream=True,
    )
    console = Console(record=True, width=70, stderr=True)
    orch.set_log_console(console=console)
    orch.queue_stream_separator()
    stream_output = io.StringIO()
    with redirect_stdout(stream_output):
        orch._stream_callback("Creating plan text")

    assert stream_output.getvalue() == "Creating plan text"
    rendered = console.export_text()
    assert "─" in rendered


def test_orchestrator_todo_calls_log_todo_state() -> None:
    orch = Orchestrator(
        context=Context(),
        executor=ToolExecutor(env={"cwd": "."}),
        model="model-x",
        tools=[],
        stream=True,
    )
    console = Console(record=True, width=90, stderr=True)
    orch.set_log_console(console=console)
    _ = orch._execute_tool_calls(
        [
            ToolCall(
                id="todo-1",
                name="todo_write",
                arguments={
                    "todos": [
                        {
                            "id": "t1",
                            "content": "Write summary",
                            "status": "in_progress",
                        },
                        {
                            "id": "t2",
                            "content": "Review output",
                            "status": "pending",
                        },
                    ]
                },
            ),
            ToolCall(id="todo-2", name="todo_read", arguments={}),
        ]
    )

    rendered = console.export_text()
    assert "[todo_write]" in rendered
    assert "[todo_read]" in rendered
    assert "[~] t1: Write summary" in rendered
    assert "[ ] t2: Review output" in rendered


def test_liteforge_plan_mode_streaming_renders_plan_when_not_streamed(
    monkeypatch,
) -> None:
    responses = iter(["no"])

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]

        @property
        def last_plan_content(self) -> str | None:
            return "Plan drafted"

        def set_log_console(self, *, console) -> None:
            del console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            self._context.add_assistant_message(content="Plan drafted", tool_calls=None)
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="plan only", task_id="lf-plan-stream"),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert "Creating plan" in rendered
    assert rendered.count("Plan drafted") == 1
    assert "╭" not in rendered


def test_liteforge_plan_mode_streaming_skips_duplicate_plan_panel(monkeypatch) -> None:
    responses = iter(["no"])

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]
            self._console = None

        @property
        def last_plan_content(self) -> str | None:
            return "Plan drafted"

        @property
        def streamed_text(self) -> str:
            return "Plan drafted"

        def set_log_console(self, *, console) -> None:
            self._console = console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            if self._console is not None:
                self._console.print("Plan drafted")
            self._context.add_assistant_message(content="Plan drafted", tool_calls=None)
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-stream-dedupe",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert rendered.count("Plan drafted") == 1


def test_liteforge_plan_mode_streaming_shows_panel_for_assistant_fallback(
    monkeypatch,
) -> None:
    responses = iter(["no"])

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]
            self._console = None

        def set_log_console(self, *, console) -> None:
            self._console = console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            if self._console is not None:
                self._console.print("Exploration text (streamed)")
            self._context.add_assistant_message(
                content="Fallback plan text",
                tool_calls=None,
            )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-stream-fallback",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert "Fallback plan text" in rendered
    assert "╭" not in rendered


def test_liteforge_plan_mode_streaming_expands_plan_path_to_contents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    responses = iter(["no"])
    plan_path = tmp_path / "generated-plan.md"
    plan_path.write_text("## Generated Plan\n\n- Step A\n- Step B\n", encoding="utf-8")

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]

        @property
        def last_plan_content(self) -> str | None:
            return f"Plan created: {plan_path}"

        def set_log_console(self, *, console) -> None:
            del console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            self._context.add_assistant_message(
                content="Plan created fallback text",
                tool_calls=None,
            )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-stream-path-content",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert "## Generated Plan" in rendered
    assert "- Step A" in rendered
    assert "- Step B" in rendered
    assert str(plan_path) not in rendered


def test_liteforge_plan_mode_streaming_materializes_wrapped_plan_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    responses = iter(["no"])
    plan_path = tmp_path / "wrapped-plan.md"
    plan_path.write_text("## Wrapped Plan\n\n- Step X\n", encoding="utf-8")

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]

        @property
        def last_plan_content(self) -> str | None:
            return f"Plan created: `{plan_path}`."

        def set_log_console(self, *, console) -> None:
            del console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-stream-wrapped-path",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert "## Wrapped Plan" in rendered
    assert "- Step X" in rendered
    assert str(plan_path) not in rendered


def test_liteforge_plan_mode_streaming_avoids_todo_write_panel_duplication(
    monkeypatch,
) -> None:
    responses = iter(["no"])

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context
            self._context = context
            self._tool_names = [tool["function"]["name"] for tool in tools]
            self._console = None
            self._visible_text = ""

        def set_log_console(self, *, console) -> None:
            self._console = console

        def queue_stream_separator(self) -> None:
            return

        @property
        def visible_text(self) -> str:
            return self._visible_text

        def run(self) -> bool:
            todo_text = "[ ] t1: Draft UI plan"
            if self._console is not None:
                self._console.print(todo_text, markup=False)
            self._visible_text = todo_text
            self._context.add_tool_result(
                ToolResult(
                    tool_call_id="todo-1",
                    name="todo_write",
                    content=todo_text,
                    is_error=False,
                )
            )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-stream-todo",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert rendered.count("[ ] t1: Draft UI plan") == 1


def test_liteforge_plan_mode_streaming_hides_review_status_when_no_plan_visible(
    monkeypatch,
) -> None:
    responses = iter(["no"])

    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self.context = context

        def set_log_console(self, *, console) -> None:
            del console

        def queue_stream_separator(self) -> None:
            return

        def run(self) -> bool:
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True
    cfg.agent_config["stream"] = True
    console = Console(record=True)

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="plan only",
            task_id="lf-plan-no-visible-content",
        ),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    rendered = console.export_text()
    assert "Plan created. Review it below." not in rendered
    assert "Plan created, but no visible plan content was captured." in rendered


def test_liteforge_stream_mode_skips_final_output_panel(monkeypatch) -> None:
    class FakeToolExecutor:
        def __init__(self, *, env: dict[str, object]) -> None:
            del env

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            context,
            executor,
            model,
            tools,
            max_requests_per_turn,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor
            del model
            del tools
            del max_requests_per_turn
            del max_tool_failure_per_turn
            del stream
            self._context = context

        def run(self) -> bool:
            self._context.add_assistant_message(
                content="streamed final response", tool_calls=None
            )
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)

    cfg = _runtime_cfg()
    cfg.agent_config["stream"] = True
    console = Console(record=True)
    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="stream output", task_id="lf-stream-1"),
        cfg=cfg,
        console=console,
        sink=None,
    )

    assert result.success
    assert result.final_message == "streamed final response"
    rendered = console.export_text()
    assert "Final Output" not in rendered
