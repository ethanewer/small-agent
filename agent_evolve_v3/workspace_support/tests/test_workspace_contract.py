# pyright: reportImplicitRelativeImport=false, reportUninitializedInstanceVariable=false, reportMissingTypeStubs=false, reportAny=false, reportUnusedCallResult=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import cast
import unittest
from unittest.mock import patch

import benchmark
from workspace_config import build_runtime_config, load_workspace_metadata

BuildHarborCommand = Callable[..., list[str]]


def _load_benchmark_expectations(
    *, repo_root: Path, script_name: str
) -> tuple[str, int]:
    script_path = repo_root / "harbor" / script_name
    dataset_ref = ""
    in_tasks = False
    task_count = 0
    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith('DATASET_REF="') and stripped.endswith('"'):
            dataset_ref = stripped[len('DATASET_REF="') : -1]
            continue
        if stripped == "BENCHMARK_TASKS=(":
            in_tasks = True
            continue
        if in_tasks and stripped == ")":
            break
        if in_tasks and stripped:
            task_count += 1
    return dataset_ref, task_count


class TestWorkspaceContract(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[1]
        self.metadata = load_workspace_metadata(workspace_root=self.workspace_root)

    def test_workspace_metadata_exists(self) -> None:
        self.assertEqual(self.metadata.baseline, "terminus2")
        self.assertTrue(self.metadata.created_from)

    def test_workspace_contains_visible_runtime_files(self) -> None:
        required = [
            "README.md",
            "pyproject.toml",
            "agents/agent.py",
            "benchmark.py",
            "workspace_harbor_agent.py",
            "model_catalog.json",
            "workspace_agent_types.py",
        ]
        required.append(f"{self.metadata.baseline}_support")
        for relative_path in required:
            self.assertTrue(
                (self.workspace_root / relative_path).exists(),
                f"Missing workspace file: {relative_path}",
            )

    def test_runtime_config_builds_for_known_model(self) -> None:
        catalog = json.loads(
            (self.workspace_root / "model_catalog.json").read_text(encoding="utf-8")
        )
        first_model_key = sorted(catalog["models"].keys())[0]
        env_name = catalog["models"][first_model_key]["api_key"]
        with patch.dict(os.environ, {env_name: "test-key"}, clear=False):
            cfg = build_runtime_config(
                workspace_root=self.workspace_root,
                model_key=first_model_key,
                final_message_enabled=False,
            )
        self.assertEqual(cfg.model.api_key, "test-key")
        self.assertFalse(cfg.agent_config["final_message"])

    def test_benchmark_command_uses_workspace_harbor_agent(self) -> None:
        dataset_ref, task_count = _load_benchmark_expectations(
            repo_root=self.workspace_root.parents[1],
            script_name="run_small_benchmark.sh",
        )
        build_harbor_command = cast(BuildHarborCommand, benchmark.build_harbor_command)
        with patch.object(benchmark, "resolve_harbor_command", return_value=["harbor"]):
            command = build_harbor_command(
                jobs_dir=self.workspace_root / "tmp-jobs",
                model_key="qwen3.5-9b",
            )
        self.assertIn("workspace_harbor_agent:WorkspaceHarborAgent", command)
        self.assertIn("--model", command)
        self.assertIn("qwen3.5-9b", command)
        self.assertIn(dataset_ref, command)
        self.assertEqual(command.count("--task-name"), task_count)

    def test_smoke_benchmark_command_uses_smoke_task_set(self) -> None:
        dataset_ref, task_count = _load_benchmark_expectations(
            repo_root=self.workspace_root.parents[1],
            script_name="run_smoke.sh",
        )
        build_harbor_command = cast(BuildHarborCommand, benchmark.build_harbor_command)
        with patch.object(benchmark, "resolve_harbor_command", return_value=["harbor"]):
            command = build_harbor_command(
                jobs_dir=self.workspace_root / "tmp-jobs",
                model_key="qwen3.5-9b",
                benchmark_preset="smoke",
            )
        self.assertIn("workspace_harbor_agent:WorkspaceHarborAgent", command)
        self.assertIn(dataset_ref, command)
        self.assertEqual(command.count("--task-name"), task_count)


if __name__ == "__main__":
    unittest.main()
