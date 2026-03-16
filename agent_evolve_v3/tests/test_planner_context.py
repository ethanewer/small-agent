# pyright: reportAny=false, reportPrivateUsage=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from agent_evolve_v3 import run_outer_loop
from agent_evolve_v3.config import RunSpec
from agent_evolve_v3.planner_context import (
    PLANNER_CONTEXT_FILE_NAME,
    PLANNER_NOTES_FILE_NAME,
    build_planner_context_bundle,
)
from agent_evolve_v3.state import AgentState, BenchmarkSummary, OfficialBenchmarkRun
from agent_evolve_v3.state_manager import StateManager


def test_build_planner_context_bundle_summarizes_deltas_and_failures(
    tmp_path: Path,
) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_benchmark(
        state=root,
        run_root=manager.run_root,
        run_id="iteration-0000",
        reward_mean=0.40,
        pass_count=8,
        failure_count=2,
        error_count=1,
        failed_trials=["timeout-task__abc"],
        exception_types={"AgentTimeoutError": ["timeout-task__abc"]},
        preview_text="agent hit timeout",
    )
    root.save()
    child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="Tighten retries and add result summaries.",
        planner_selected_state_index=0,
        planner_prompt_artifact_path=manager.run_root
        / "artifacts"
        / "planner_prompt.txt",
        planner_output_artifact_path=manager.run_root
        / "artifacts"
        / "planner_output.txt",
    )
    _attach_benchmark(
        state=child,
        run_root=manager.run_root,
        run_id="iteration-0001",
        reward_mean=0.55,
        pass_count=11,
        failure_count=1,
        error_count=0,
        failed_trials=["regression-task__def"],
        exception_types={},
        preview_text="verifier expected localhost response",
    )
    child.notes = "Iteration notes\n- Added tighter retries\n"
    child.save()

    bundle = build_planner_context_bundle(
        states=manager.states,
        run_root=manager.run_root,
    )

    assert bundle.planning_context_payload["iteration_count"] == 2
    assert bundle.planning_context_payload["best_completed_iteration"] == 1
    assert "reward +0.150" in bundle.score_analysis_text
    assert "timeout-task__abc" in bundle.failed_task_trajectories_text
    assert "Tighten retries and add result summaries." in bundle.plan_history_text
    failed_iterations = bundle.failed_task_index_payload[
        "iterations_with_problem_tasks"
    ]
    assert len(failed_iterations) == 2


def test_planning_environment_exports_full_context_and_notes(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_benchmark(
        state=root,
        run_root=manager.run_root,
        run_id="iteration-0000",
        reward_mean=0.45,
        pass_count=9,
        failure_count=1,
        error_count=0,
        failed_trials=["broken-task__xyz"],
        exception_types={"AgentTimeoutError": ["broken-task__xyz"]},
        preview_text="agent ran out of time",
    )
    root.save()
    failed_child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="Try a brittle follow-up.",
        planner_selected_state_index=0,
        planner_prompt_artifact_path=manager.run_root
        / "artifacts"
        / "planner_prompt.txt",
        planner_output_artifact_path=manager.run_root
        / "artifacts"
        / "planner_output.txt",
    )
    (manager.run_root / "artifacts" / "iteration-0001").mkdir(
        parents=True, exist_ok=True
    )
    (
        manager.run_root / "artifacts" / "iteration-0001" / "implementation_step.json"
    ).write_text(
        json.dumps(
            {"returncode": 2, "stdout": "boom", "stderr": ""}, ensure_ascii=True
        ),
        encoding="utf-8",
    )
    notes_path = manager.run_root / PLANNER_NOTES_FILE_NAME
    notes_path.write_text("# Planner Notes\n\nseeded\n", encoding="utf-8")

    with manager.planning_environment(
        scoreboard_text="## Scoreboard\n",
        planner_notes_path=notes_path,
    ) as (planning_root, candidate_states):
        candidate_payload = json.loads(
            (planning_root / "states.json").read_text(encoding="utf-8")
        )
        planning_context = json.loads(
            (planning_root / PLANNER_CONTEXT_FILE_NAME).read_text(encoding="utf-8")
        )
        copied_notes = (planning_root / PLANNER_NOTES_FILE_NAME).read_text(
            encoding="utf-8"
        )
        failed_task_text = (planning_root / "failed_task_trajectories.md").read_text(
            encoding="utf-8"
        )
        plan_history_text = (planning_root / "plan_history.md").read_text(
            encoding="utf-8"
        )

    assert len(candidate_states) == 1
    assert len(candidate_payload) == 1
    assert planning_context["iteration_count"] == 2
    assert planning_context["iterations"][1]["status"] == "implementation_failed"
    assert copied_notes == "# Planner Notes\n\nseeded\n"
    assert "broken-task__xyz" in failed_task_text
    assert failed_child.plan is not None
    assert failed_child.plan in plan_history_text


def test_prepare_planner_notes_seeds_latest_iteration_once(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_benchmark(
        state=root,
        run_root=manager.run_root,
        run_id="iteration-0000",
        reward_mean=0.35,
        pass_count=7,
        failure_count=3,
        error_count=0,
        failed_trials=["failing-task__abc"],
        exception_types={},
        preview_text="preview text",
    )
    root.save()

    notes_path = run_outer_loop._prepare_planner_notes(
        run_root=manager.run_root,
        states=manager.states,
    )
    run_outer_loop._prepare_planner_notes(
        run_root=manager.run_root,
        states=manager.states,
    )

    content = notes_path.read_text(encoding="utf-8")
    assert content.count("## Iteration 0") == 1
    assert "### Result" in content
    assert "Problem tasks to inspect: failing-task__abc" in content


def test_persist_failed_state_context_captures_workspace_notes(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    child = manager.create_child_state(
        parent_state=root,
        iteration=1,
        plan="Persist notes after failure.",
        planner_selected_state_index=0,
        planner_prompt_artifact_path=manager.run_root
        / "artifacts"
        / "planner_prompt.txt",
        planner_output_artifact_path=manager.run_root
        / "artifacts"
        / "planner_output.txt",
    )
    notes_path = Path(child.refiner_workspace_path) / "NOTES.md"
    notes_path.write_text("Failure notes\n- validation exploded\n", encoding="utf-8")

    run_outer_loop._persist_failed_state_context(
        manager=manager,
        run_root=manager.run_root,
        state=child,
    )

    reloaded = AgentState.load(path=Path(child.path))
    assert reloaded.notes == "Failure notes\n- validation exploded\n"
    assert (manager.run_root / "SCOREBOARD.md").exists()


def test_render_planner_prompt_mentions_new_inputs(tmp_path: Path) -> None:
    manager, root = _bootstrap_manager(tmp_path=tmp_path)
    _attach_benchmark(
        state=root,
        run_root=manager.run_root,
        run_id="iteration-0000",
        reward_mean=0.45,
        pass_count=9,
        failure_count=2,
        error_count=1,
        failed_trials=["latest-task__123"],
        exception_types={"AgentTimeoutError": ["latest-task__123"]},
        preview_text="timeout preview",
    )
    root.save()

    prompt = run_outer_loop._render_planner_prompt(
        run_root=manager.run_root,
        latest_state=root,
        best_completed_state=root,
        scoreboard_text="## Scoreboard\n",
        benchmark_model_key=manager.run_spec.model_key,
        candidate_state_count=1,
        iteration_count=1,
    )

    assert "planning_context.json" in prompt
    assert "score_analysis.md" in prompt
    assert "failed_task_trajectories.md" in prompt
    assert "PLANNER_NOTES.md" in prompt
    assert "latest-task__123" in prompt


def _bootstrap_manager(*, tmp_path: Path) -> tuple[StateManager, AgentState]:
    repo_root = Path(__file__).resolve().parents[2]
    run_root = tmp_path / "run"
    run_spec = RunSpec(
        name="demo",
        baseline="terminus2",
        model_key="qwen3.5-9b",
        cursor_model="gpt-5.3-codex-high",
        iterations=3,
        random_seed=7,
        benchmark_tasks=(),
    )
    manager = StateManager(
        repo_root=repo_root,
        run_root=run_root,
        run_spec=run_spec,
    )
    root = manager.bootstrap_root_state()
    return manager, root


def _attach_benchmark(
    *,
    state: AgentState,
    run_root: Path,
    run_id: str,
    reward_mean: float,
    pass_count: int,
    failure_count: int,
    error_count: int,
    failed_trials: list[str],
    exception_types: dict[str, list[str]],
    preview_text: str,
) -> None:
    artifacts_dir = run_root / "artifacts" / f"iteration-{state.iteration:04d}"
    benchmark_dir = artifacts_dir / "official_benchmark"
    harbor_job_dir = benchmark_dir / "harbor_job"
    harbor_job_dir.mkdir(parents=True, exist_ok=True)
    aggregate_result_path = harbor_job_dir / "result.json"
    aggregate_result_path.write_text(
        json.dumps(
            {
                "stats": {
                    "n_trials": pass_count + failure_count + error_count,
                    "n_errors": error_count,
                    "evals": {
                        "small-agent-harbor-external__terminal-bench": {
                            "n_trials": pass_count + failure_count + error_count,
                            "n_errors": error_count,
                            "metrics": [{"mean": reward_mean}],
                            "reward_stats": {
                                "reward": {
                                    "1.0": [f"pass-task__{run_id}"],
                                    "0.0": failed_trials,
                                }
                            },
                            "exception_stats": exception_types,
                        }
                    },
                }
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    for trial_id in failed_trials:
        trial_dir = harbor_job_dir / trial_id
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": trial_id.split("__", maxsplit=1)[0],
                    "agent_result": {
                        "metadata": {
                            "small_agent_result": {
                                "stdout": preview_text,
                                "stderr": "",
                            }
                        }
                    },
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        (trial_dir / "agent" / "trajectory.json").write_text(
            json.dumps({"steps": ["inspect", "retry"]}, ensure_ascii=True),
            encoding="utf-8",
        )
        (trial_dir / "verifier" / "test-stdout.txt").write_text(
            f"verifier: {preview_text}",
            encoding="utf-8",
        )
    summary_path = benchmark_dir / "benchmark_summary.json"
    summary = BenchmarkSummary(
        created_at_utc="now",
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        reward_mean=reward_mean,
        n_trials=pass_count + failure_count + error_count,
        pass_count=pass_count,
        failure_count=failure_count,
        error_count=error_count,
        passed_trials=[f"pass-task__{run_id}"],
        failed_trials=failed_trials,
        exception_types=exception_types,
    )
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    stdout_path = benchmark_dir / "benchmark_stdout.log"
    stderr_path = benchmark_dir / "benchmark_stderr.log"
    stdout_path.write_text("stdout\n", encoding="utf-8")
    stderr_path.write_text("stderr\n", encoding="utf-8")
    (artifacts_dir / "benchmark_step.json").write_text(
        json.dumps({"returncode": 0, "stdout": "", "stderr": ""}, ensure_ascii=True),
        encoding="utf-8",
    )
    state.official_benchmark = OfficialBenchmarkRun(
        model_key="qwen3.5-9b",
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        benchmark_summary_path=str(summary_path),
        benchmark_stdout_path=str(stdout_path),
        benchmark_stderr_path=str(stderr_path),
        created_at_utc="now",
    )
    state.result = summary
