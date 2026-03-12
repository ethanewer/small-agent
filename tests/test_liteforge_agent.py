from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from agents.core.task import Task
from agents.interface import AgentModelConfig, AgentRuntimeConfig
from agents.liteforge.context import Context
from agents.liteforge.runtime_agent import LiteforgeAgent
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
    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="say hello", task_id="lf-1"),
        cfg=cfg,
        console=Console(record=True),
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
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg(cwd=str(tmp_path))
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(
            instruction="implement feature", task_id="lf-plan-1"
        ),
        cfg=cfg,
        console=Console(record=True),
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


def test_liteforge_plan_mode_rejects_without_execution_phase(monkeypatch) -> None:
    recorded_tool_name_sequences: list[list[str]] = []
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
            tool_names = [tool["function"]["name"] for tool in tools]
            recorded_tool_name_sequences.append(tool_names)
            self.context = context
            self._context = context

        def run(self) -> bool:
            self._context.add_assistant_message(content="Plan drafted", tool_calls=None)
            return True

    monkeypatch.setattr("agents.liteforge.runtime_agent.ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr("agents.liteforge.runtime_agent.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    cfg = _runtime_cfg()
    cfg.agent_config["plan_mode"] = True
    cfg.agent_config["readonly"] = True

    result = LiteforgeAgent().run_task(
        task=Task.from_instruction(instruction="plan only", task_id="lf-plan-2"),
        cfg=cfg,
        console=Console(record=True),
        sink=None,
    )

    assert result.success
    assert result.exit_code == 0
    assert result.final_message == "Plan drafted"
    assert recorded_tool_name_sequences == [list(READONLY_TOOL_NAMES)]
