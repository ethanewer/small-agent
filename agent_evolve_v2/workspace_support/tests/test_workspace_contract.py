# pyright: reportImplicitRelativeImport=false, reportUninitializedInstanceVariable=false, reportMissingTypeStubs=false, reportAny=false, reportUnusedCallResult=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import benchmark
from workspace_config import build_runtime_config, load_workspace_metadata


class TestWorkspaceContract(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[1]
        self.metadata = load_workspace_metadata(workspace_root=self.workspace_root)

    def test_workspace_metadata_exists(self) -> None:
        self.assertIn(self.metadata.baseline, {"liteforge", "terminus2"})
        self.assertTrue(self.metadata.created_from)

    def test_workspace_contains_visible_runtime_files(self) -> None:
        required = [
            "README.md",
            "NOTES.md",
            "pyproject.toml",
            "agents/agent.py",
            "benchmark.py",
            "benchmark_cache.py",
            "workspace_harbor_agent.py",
            "run_task.py",
            "validate_workspace.py",
            "critic_tools.py",
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
        with patch.object(benchmark, "resolve_harbor_command", return_value=["harbor"]):
            command = benchmark.build_harbor_command(
                jobs_dir=self.workspace_root / "tmp-jobs",
                model_key="qwen3.5-9b",
            )
        self.assertIn("workspace_harbor_agent:WorkspaceHarborAgent", command)
        self.assertIn("--model", command)
        self.assertIn("qwen3.5-9b", command)
        self.assertGreaterEqual(command.count("--task-name"), 20)


if __name__ == "__main__":
    unittest.main()
