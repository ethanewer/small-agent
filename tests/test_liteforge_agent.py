from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

from rich.console import Console

from agents.core.task import Task
from agents.interface import AgentModelConfig, AgentRuntimeConfig
from agents.liteforge.context import Context, ToolCall
from agents.liteforge.orchestrator import Orchestrator
from agents.liteforge.runtime_agent import LiteforgeAgent
from agents.liteforge.tools.executor import ToolExecutor
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


def test_orchestrator_trims_tool_log_detail_newlines() -> None:
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
    console = Console(record=True, width=80, stderr=True)
    orch.set_log_console(console=console)
    _ = orch._execute_tool_calls(
        [
            ToolCall(
                id="tool-1",
                name="shell",
                arguments={"description": "List files\n", "command": "ls"},
            ),
            ToolCall(
                id="tool-2",
                name="read",
                arguments={"file_path": "/tmp/demo.py\n"},
            ),
        ]
    )

    rendered = console.export_text()
    assert "[shell] List files\n\n" not in rendered
    assert "[read] /tmp/demo.py\n\n" not in rendered
    assert "[shell] List files" in rendered
    assert "[read] /tmp/demo.py" in rendered


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


def test_orchestrator_ignores_whitespace_only_stream_before_tool_logs() -> None:
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
                    arguments={"description": "List files", "command": "ls"},
                )
            ]
        )
        orch._stream_callback("\n\n")
        _ = orch._execute_tool_calls(
            [
                ToolCall(
                    id="tool-2",
                    name="read",
                    arguments={"file_path": "/tmp/demo.py"},
                )
            ]
        )

    assert stream_output.getvalue() == ""
    rendered = console.export_text()
    assert "[shell] List files\n\n\n─" not in rendered
    assert "[read] /tmp/demo.py" in rendered


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
