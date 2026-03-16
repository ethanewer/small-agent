# pyright: reportImplicitRelativeImport=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingImports=false, reportAny=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from workspace_agent_types import WorkspaceModelConfig, WorkspaceRuntimeConfig
from workspace_config import load_workspace_metadata


def test_baseline_runtime_smoke() -> None:
    metadata = load_workspace_metadata(
        workspace_root=Path(__file__).resolve().parents[1]
    )
    assert metadata.baseline == "terminus2"
    _assert_terminus2_runtime()


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
