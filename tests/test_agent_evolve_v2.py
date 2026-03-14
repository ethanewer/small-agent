# pyright: reportAny=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_evolve_v2.benchmark import resolve_benchmark_cache_root
from agent_evolve_v2.config import RunSpec, load_runs_config
from agent_evolve_v2.critic import summarize_job
from agent_evolve_v2.state import AgentState, BenchmarkSummary, OfficialBenchmarkRun
from agent_evolve_v2.state_manager import StateManager
from agent_evolve_v2.workspace_support.benchmark_cache import (
    CanonicalRunRecord,
    compute_benchmark_fingerprint,
    resolve_latest_visible_run_dir,
    write_canonical_run,
)
from agent_evolve_v2.workspace_support.workspace_harbor_agent import (
    WorkspaceHarborAgent,
)


EXPECTED_WORKDIR_ENTRIES = {
    "README.md",
    "NOTES.md",
    "agent",
    "run_benchmark.sh",
    "test_agent.sh",
}


def test_load_runs_config_supports_json_subset_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "runs.json"
    config_path.write_text(
        json.dumps(
            {
                "default_run": "demo",
                "runs": {
                    "demo": {
                        "baseline": "liteforge",
                        "model_key": "qwen3.5-flash",
                        "cursor_model": "opus-4.6-thinking",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_runs_config(path=config_path)
    run = loaded.get_run(run_name=None)
    assert run.name == "demo"
    assert run.baseline == "liteforge"


@pytest.mark.parametrize("baseline", ["liteforge", "terminus2"])
def test_state_manager_bootstraps_minimal_workspace(
    tmp_path: Path, baseline: str
) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path, baseline=baseline)
    workspace = Path(root.refiner_workspace_path)
    assert set(_visible_names(workspace=workspace)) == EXPECTED_WORKDIR_ENTRIES
    assert (workspace / "agent" / "agent.py").exists()
    assert (workspace / "agent" / "runtime_types.py").exists()
    assert (workspace / "test_agent.sh").exists()
    assert (workspace / "run_benchmark.sh").exists()
    assert manager.run_spec.model_key in (workspace / "README.md").read_text(
        encoding="utf-8"
    )
    assert manager.run_spec.model_key in (workspace / "NOTES.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("baseline", ["liteforge", "terminus2"])
def test_start_workdirs_are_preassembled_and_minimal(baseline: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    start_dir = repo_root / "agent_evolve_v2" / "start_workdirs" / baseline
    assert set(_visible_names(workspace=start_dir)) == EXPECTED_WORKDIR_ENTRIES
    assert (start_dir / "agent" / "agent.py").exists()
    assert (start_dir / "agent" / "runtime_types.py").exists()
    assert "service_cli" not in (start_dir / "README.md").read_text(encoding="utf-8")
    assert "./test_agent.sh" in (start_dir / "README.md").read_text(encoding="utf-8")
    assert "./run_benchmark.sh" in (start_dir / "README.md").read_text(encoding="utf-8")


def test_state_manager_uses_thompson_sample_scores(tmp_path: Path) -> None:
    class FakeRandom:
        def __init__(self) -> None:
            self.samples = iter([0.1, 0.9])

        def betavariate(self, alpha: float, beta: float) -> float:
            del alpha, beta
            return next(self.samples)

    manager, _root = _bootstrap_manager(tmp_path=tmp_path)
    first = AgentState(
        prev_path=None,
        path=str((manager.states_dir / "iteration-0001.json").resolve()),
        refiner_workspace_path=str(
            (manager.workspaces_dir / "iteration-0001").resolve()
        ),
        iteration=1,
        baseline="liteforge",
        result=BenchmarkSummary(
            created_at_utc="now",
            aggregate_result_path="a",
            harbor_job_dir="a",
            reward_mean=0.2,
            n_trials=5,
            pass_count=1,
            failure_count=4,
            error_count=0,
        ),
    )
    second = AgentState(
        prev_path=None,
        path=str((manager.states_dir / "iteration-0002.json").resolve()),
        refiner_workspace_path=str(
            (manager.workspaces_dir / "iteration-0002").resolve()
        ),
        iteration=2,
        baseline="liteforge",
        result=BenchmarkSummary(
            created_at_utc="now",
            aggregate_result_path="b",
            harbor_job_dir="b",
            reward_mean=0.8,
            n_trials=5,
            pass_count=4,
            failure_count=1,
            error_count=0,
        ),
    )
    first.save()
    second.save()
    sampled = manager.sample_parent_state(rng=FakeRandom())
    assert sampled.state.iteration == 2
    assert sampled.alpha == 5.0
    assert sampled.beta == 2.0


def test_child_workspace_ignores_transient_runtime_dirs(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    parent_workspace = Path(root.refiner_workspace_path)
    (parent_workspace / ".venv").mkdir()
    (parent_workspace / ".venv" / "marker.txt").write_text(
        "transient\n", encoding="utf-8"
    )
    (parent_workspace / "benchmark-artifacts").mkdir()
    (parent_workspace / "benchmark-artifacts" / "marker.txt").write_text(
        "transient\n",
        encoding="utf-8",
    )
    (parent_workspace / "outputs").mkdir()
    (parent_workspace / "outputs" / "marker.txt").write_text(
        "visible\n", encoding="utf-8"
    )

    child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="check transient ignores",
    )
    child_workspace = Path(child.refiner_workspace_path)
    assert not (child_workspace / ".venv").exists()
    assert not (child_workspace / "benchmark-artifacts").exists()
    assert not (child_workspace / "outputs").exists()


def test_child_workspace_rewrites_model_key_in_docs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "run"
    run_spec = RunSpec(
        name="demo",
        baseline="liteforge",
        model_key="qwen3.5-9b",
        cursor_model="gpt-5.3-codex-high",
    )
    manager = StateManager(repo_root=repo_root, run_root=run_root, run_spec=run_spec)
    root = manager.bootstrap_root_state()
    parent_workspace = Path(root.refiner_workspace_path)
    for relative_path in ("README.md", "NOTES.md"):
        doc_path = parent_workspace / relative_path
        doc_path.write_text(
            doc_path.read_text(encoding="utf-8")
            .replace(run_spec.model_key, "qwen3.5-flash")
            .replace("./test_agent.sh qwen3.5-9b", "./test_agent.sh qwen3.5-flash")
            .replace(
                "./run_benchmark.sh qwen3.5-9b",
                "./run_benchmark.sh qwen3.5-flash",
            ),
            encoding="utf-8",
        )

    child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="rewrite docs",
    )
    child_workspace = Path(child.refiner_workspace_path)
    assert "./test_agent.sh qwen3.5-9b" in (child_workspace / "README.md").read_text(
        encoding="utf-8"
    )
    assert "./run_benchmark.sh qwen3.5-9b" in (child_workspace / "NOTES.md").read_text(
        encoding="utf-8"
    )


def test_summarize_job_extracts_summary_and_critic(tmp_path: Path) -> None:
    harbor_job_dir = tmp_path / "job-1"
    harbor_job_dir.mkdir()
    (harbor_job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_trials": 2,
                    "n_errors": 1,
                    "evals": {
                        "workspace-agent__terminal-bench": {
                            "n_trials": 2,
                            "n_errors": 1,
                            "metrics": [{"mean": 0.5}],
                            "reward_stats": {
                                "reward": {
                                    "1.0": ["good-task__abc"],
                                    "0.0": ["bad-task__xyz"],
                                }
                            },
                            "exception_stats": {"AgentTimeoutError": ["bad-task__xyz"]},
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    failed_trial = harbor_job_dir / "bad-task__xyz"
    (failed_trial / "verifier").mkdir(parents=True)
    (failed_trial / "agent").mkdir(parents=True)
    (failed_trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "bad-task",
                "agent_result": {
                    "metadata": {
                        "small_agent_result": {
                            "stdout": "runtime budget exhaustion",
                            "stderr": "",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (failed_trial / "verifier" / "test-stdout.txt").write_text(
        "Verifier expected localhost to respond",
        encoding="utf-8",
    )
    (failed_trial / "agent" / "trajectory.json").write_text(
        '{"steps": ["explore", "retry", "retry"]}',
        encoding="utf-8",
    )

    summary, critic = summarize_job(
        harbor_job_dir=harbor_job_dir,
        max_failure_items=3,
        max_success_items=1,
    )
    assert summary.pass_count == 1
    assert summary.failure_count == 1
    assert summary.error_count == 1
    assert critic.items
    assert critic.items[0].failure_type == "timeout"


def test_workspace_harbor_agent_applies_extra_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    WorkspaceHarborAgent(extra_env={"OPENROUTER_API_KEY": "test-key"})
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"


def test_benchmark_fingerprint_ignores_notes_and_readme_edits(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_minimal_agent_tree(workspace=workspace)
    (workspace / "README.md").write_text("readme v1\n", encoding="utf-8")
    (workspace / "NOTES.md").write_text("notes v1\n", encoding="utf-8")
    before = compute_benchmark_fingerprint(
        workspace_root=workspace,
        model_key="qwen3.5-9b",
    )
    (workspace / "README.md").write_text("readme v2\n", encoding="utf-8")
    (workspace / "NOTES.md").write_text("notes v2\n", encoding="utf-8")
    (workspace / "run_benchmark.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    after = compute_benchmark_fingerprint(
        workspace_root=workspace,
        model_key="qwen3.5-9b",
    )
    assert before.fingerprint == after.fingerprint

    prompt_path = workspace / "agent" / "templates" / "forge.md"
    prompt_path.write_text("prompt v2\n", encoding="utf-8")
    prompt_changed = compute_benchmark_fingerprint(
        workspace_root=workspace,
        model_key="qwen3.5-9b",
    )
    assert prompt_changed.fingerprint != after.fingerprint


def test_benchmark_fingerprint_is_stable_across_workspace_copies(
    tmp_path: Path,
) -> None:
    first = tmp_path / "iteration-0001"
    second = tmp_path / "iteration-0002"
    _write_minimal_agent_tree(workspace=first)
    for path in sorted(first.rglob("*")):
        if path.is_dir():
            continue
        target = second / path.relative_to(first)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    fingerprint_one = compute_benchmark_fingerprint(
        workspace_root=first,
        model_key="qwen3.5-9b",
    )
    fingerprint_two = compute_benchmark_fingerprint(
        workspace_root=second,
        model_key="qwen3.5-9b",
    )
    assert fingerprint_one.fingerprint == fingerprint_two.fingerprint


def test_resolve_benchmark_cache_root_defaults_outside_workspace(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "agent_evolve_v2" / "start_workdirs" / "liteforge"
    workspace.mkdir(parents=True)
    (repo_root / "agent_evolve_v2" / "runs.json").write_text("{}", encoding="utf-8")
    cache_root = resolve_benchmark_cache_root(workspace_root=workspace)
    assert cache_root == (
        repo_root / "agent_evolve_v2" / "outputs" / "manual_benchmark_cache"
    )
    assert not str(cache_root).startswith(str(workspace))


def test_child_workspace_starts_with_single_visible_official_run(
    tmp_path: Path,
) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_official_benchmark(manager=manager, state=root, run_id="seed-run")
    manager.seed_refiner_outputs(state=root)
    root_workspace = Path(root.refiner_workspace_path)
    (root_workspace / "outputs" / "benchmark_runs" / "extra-run").mkdir(parents=True)

    child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="copy from parent",
    )
    child_workspace = Path(child.refiner_workspace_path)
    runs_root = child_workspace / "outputs" / "benchmark_runs"
    visible_runs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(visible_runs) == 1
    assert visible_runs[0] == resolve_latest_visible_run_dir(
        workspace_root=child_workspace
    )


def test_create_critic_workspace_trims_outputs_to_single_run(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_official_benchmark(manager=manager, state=root, run_id="seed-run")
    manager.seed_refiner_outputs(state=root)
    root_workspace = Path(root.refiner_workspace_path)
    extra_run = root_workspace / "outputs" / "benchmark_runs" / "extra-run"
    extra_run.mkdir(parents=True)
    (extra_run / "benchmark_summary.json").write_text("{}", encoding="utf-8")

    critic_workspace = manager.create_critic_workspace(state=root)
    visible_runs = [
        path
        for path in (critic_workspace / "outputs" / "benchmark_runs").iterdir()
        if path.is_dir()
    ]
    assert len(visible_runs) == 1
    assert root.critic_workspace_path == str(critic_workspace.resolve())


def _bootstrap_manager(
    *,
    tmp_path: Path,
    baseline: str = "liteforge",
) -> tuple[StateManager, AgentState]:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "run"
    run_spec = RunSpec(
        name="demo",
        baseline=baseline,
        model_key="qwen3.5-9b",
        cursor_model="gpt-5.3-codex-high",
    )
    manager = StateManager(repo_root=repo_root, run_root=run_root, run_spec=run_spec)
    root = manager.bootstrap_root_state()
    return manager, root


def _attach_official_benchmark(
    *,
    manager: StateManager,
    state: AgentState,
    run_id: str,
) -> None:
    run_dir = manager.benchmark_cache_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "benchmark_summary.json"
    stdout_path = run_dir / "benchmark_stdout.log"
    stderr_path = run_dir / "benchmark_stderr.log"
    aggregate_result_path = run_dir / "result.json"
    harbor_job_dir = run_dir / "harbor_jobs" / "job-1"
    harbor_job_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "created_at_utc": "now",
        "aggregate_result_path": str(aggregate_result_path),
        "harbor_job_dir": str(harbor_job_dir),
        "reward_mean": 0.5,
        "n_trials": 2,
        "pass_count": 1,
        "failure_count": 1,
        "error_count": 0,
        "passed_trials": [],
        "failed_trials": [],
        "exception_types": {},
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    stdout_path.write_text("stdout\n", encoding="utf-8")
    stderr_path.write_text("stderr\n", encoding="utf-8")
    aggregate_result_path.write_text("{}", encoding="utf-8")
    fingerprint = "f" * 64
    record = CanonicalRunRecord(
        run_id=run_id,
        fingerprint=fingerprint,
        model_key="qwen3.5-9b",
        cache_root=str(manager.benchmark_cache_dir),
        canonical_run_dir=str(run_dir),
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        benchmark_summary_path=str(summary_path),
        benchmark_stdout_path=str(stdout_path),
        benchmark_stderr_path=str(stderr_path),
        created_at_utc="now",
        status="completed",
        included_files=["agent/agent.py"],
    )
    write_canonical_run(cache_root=manager.benchmark_cache_dir, record=record)
    state.official_benchmark = OfficialBenchmarkRun(
        fingerprint=fingerprint,
        model_key="qwen3.5-9b",
        canonical_run_dir=str(run_dir),
        canonical_manifest_path=str(run_dir / "run_manifest.json"),
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        benchmark_summary_path=str(summary_path),
        benchmark_stdout_path=str(stdout_path),
        benchmark_stderr_path=str(stderr_path),
        included_files=["agent/agent.py"],
        cache_hit=True,
        created_at_utc="now",
    )


def _visible_names(*, workspace: Path) -> list[str]:
    return sorted(path.name for path in workspace.iterdir())


def _write_minimal_agent_tree(*, workspace: Path) -> None:
    (workspace / "agent" / "templates").mkdir(parents=True)
    (workspace / "agent" / "agent.py").write_text(
        "from runtime_types import WorkspaceRunResult, WorkspaceRuntimeConfig\n",
        encoding="utf-8",
    )
    (workspace / "agent" / "runtime_types.py").write_text(
        (
            "from dataclasses import dataclass, field\n"
            "from typing import Any\n"
            "@dataclass(frozen=True)\n"
            "class WorkspaceModelConfig:\n"
            "    model: str\n"
            "    api_base: str\n"
            "    api_key: str\n"
            "    temperature: float | None = None\n"
            "    context_length: int | None = None\n"
            "    extra_params: dict[str, Any] | None = None\n"
            "@dataclass(frozen=True)\n"
            "class WorkspaceRuntimeConfig:\n"
            "    model: WorkspaceModelConfig\n"
            "    agent_config: dict[str, Any] = field(default_factory=dict)\n"
        ),
        encoding="utf-8",
    )
    (workspace / "agent" / "templates" / "forge.md").write_text(
        "prompt v1\n",
        encoding="utf-8",
    )
