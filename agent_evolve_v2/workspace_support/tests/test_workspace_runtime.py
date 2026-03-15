# pyright: reportImplicitRelativeImport=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingImports=false, reportAny=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from workspace_agent_types import WorkspaceModelConfig, WorkspaceRuntimeConfig
from workspace_config import load_workspace_metadata


def test_baseline_runtime_smoke(tmp_path: Path, monkeypatch) -> None:
    metadata = load_workspace_metadata(
        workspace_root=Path(__file__).resolve().parents[1]
    )
    if metadata.baseline == "liteforge":
        _assert_liteforge_runtime(tmp_path=tmp_path, monkeypatch=monkeypatch)
    else:
        _assert_terminus2_runtime()


def test_liteforge_tool_roundtrip(tmp_path: Path) -> None:
    metadata = load_workspace_metadata(
        workspace_root=Path(__file__).resolve().parents[1]
    )
    if metadata.baseline != "liteforge":
        return

    from liteforge_support.tools import fs_patch, fs_read, fs_remove, fs_undo, fs_write

    env = {
        "cwd": str(tmp_path),
        "shell": "/bin/sh",
        "max_read_size": 2000,
    }
    snapshots: dict[str, str | None] = {}
    created = fs_write.execute(
        args={"file_path": "notes.txt", "content": "hello world"},
        env=env,
        snapshots=snapshots,
    )
    assert "Created file:" in created

    read_result = fs_read.execute(args={"file_path": "notes.txt"}, env=env)
    assert "1:hello world" in read_result

    patched = fs_patch.execute(
        args={
            "file_path": "notes.txt",
            "old_string": "hello world",
            "new_string": "goodbye world",
        },
        env=env,
        snapshots=snapshots,
    )
    assert "Replaced 1 occurrence" in patched

    removed = fs_remove.execute(
        args={"path": "notes.txt"},
        env=env,
        snapshots=snapshots,
    )
    assert "Removed file:" in removed

    restored = fs_undo.execute(
        args={"path": "notes.txt"},
        env=env,
        snapshots=snapshots,
    )
    assert "to previous state" in restored


def _assert_liteforge_runtime(*, tmp_path: Path, monkeypatch) -> None:
    module = _load_workspace_agent_module()
    Context = module.Context
    WorkspaceAgent = module.WorkspaceAgent
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
            max_turns,
            max_tool_failure_per_turn,
            stream,
        ) -> None:
            del executor, tools
            recorded["model"] = model
            recorded["max_turns"] = max_turns
            recorded["max_tool_failure_per_turn"] = max_tool_failure_per_turn
            recorded["stream"] = stream
            recorded["context"] = context

        def run(self) -> bool:
            context = recorded["context"]
            assert isinstance(context, Context)
            context.add_assistant_message(content="liteforge complete", tool_calls=None)
            return True

    monkeypatch.setattr(module, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(module, "Orchestrator", FakeOrchestrator)

    cfg = WorkspaceRuntimeConfig(
        model=WorkspaceModelConfig(
            model="openai/gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
        ),
        agent_config={
            "stream": False,
            "cwd": str(tmp_path),
            "max_turns": 4,
            "max_tool_failure_per_turn": 2,
        },
    )
    result = WorkspaceAgent().run_task(
        instruction="say hello",
        cfg=cfg,
        console=Console(record=True),
        task_id="lf-1",
    )
    assert result.success
    assert result.final_message == "liteforge complete"
    assert recorded["max_turns"] == 4


def _assert_terminus2_runtime() -> None:
    module = _load_workspace_agent_module()
    WorkspaceAgent = module.WorkspaceAgent
    cfg = WorkspaceRuntimeConfig(
        model=WorkspaceModelConfig(
            model="model-y",
            api_base="https://example.invalid/v1",
            api_key="api",
        ),
        agent_config={"max_turns": 1, "max_wait_seconds": 1.0},
    )
    with patch.object(module, "run_agent", return_value=0):
        result = WorkspaceAgent().run_task(
            instruction="inspect",
            cfg=cfg,
            console=Console(record=True),
            task_id="t2",
        )
    assert result.success
    assert result.task_id == "t2"


def _load_workspace_agent_module():
    workspace_root = Path(__file__).resolve().parents[1]
    module_path = workspace_root / "agents" / "agent.py"
    spec = importlib.util.spec_from_file_location(
        name="workspace_agent_for_tests",
        location=module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
