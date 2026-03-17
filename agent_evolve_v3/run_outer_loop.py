# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys

from agent_evolve_v3.config import RunSpec, load_runs_config
from agent_evolve_v3.prompts import load_implementation_prompt, load_planning_prompt
from agent_evolve_v3.services.benchmark import (
    load_workspace_benchmark_result,
    prepull_task_images,
    run_workspace_benchmark,
)
from agent_evolve_v3.services.runtime import (
    record_completed_process,
    run_implementation_agent,
    run_planner_agent,
    run_workspace_validation,
)
from agent_evolve_v3.state import AgentState, PlanningOutput
from agent_evolve_v3.state.manager import StateManager
from agent_evolve_v3.state.planner_context import (
    PLANNER_NOTES_FILE_NAME,
    classify_state_status,
    latest_iteration_header,
    latest_iteration_section,
    latest_run_artifact_map,
    latest_run_parent_iteration_text,
    latest_run_selectable_text,
    plan_summary,
    planner_notes_template,
    summarize_problem_trials,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-contained agent_evolve_v3 loop.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent_evolve_v3/runs.json"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "agent_evolve_v3" / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if args.resume:
        run_root = args.resume.resolve()
        manifest = json.loads(
            (run_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        run_spec = RunSpec(
            name=str(manifest["name"]),
            baseline=str(manifest["baseline"]),
            model_key=str(manifest["model_key"]),
            cursor_model=str(manifest["cursor_model"]),
            iterations=int(manifest["iterations"]),
            random_seed=int(manifest["random_seed"]),
            benchmark_tasks=tuple(
                str(task_name) for task_name in manifest.get("benchmark_tasks", [])
            ),
        )
    else:
        runs_config = load_runs_config(path=(repo_root / args.config).resolve())
        run_spec = runs_config.get_run(run_name=args.run_name)
        if args.iterations is not None:
            run_spec = RunSpec(
                name=run_spec.name,
                baseline=run_spec.baseline,
                model_key=run_spec.model_key,
                cursor_model=run_spec.cursor_model,
                iterations=max(1, args.iterations),
                random_seed=run_spec.random_seed,
                benchmark_tasks=run_spec.benchmark_tasks,
            )
        run_root = _create_run_root(outputs_root=outputs_root, run_spec=run_spec)

    manager = StateManager(
        repo_root=repo_root,
        run_root=run_root,
        run_spec=run_spec,
    )
    _write_manifest(run_root=run_root, run_spec=run_spec)
    prepull_task_images(
        task_names=run_spec.benchmark_tasks,
        repo_root=repo_root,
    )

    root_state = manager.bootstrap_root_state()
    try:
        _ensure_state_evaluated(
            manager=manager,
            run_root=run_root,
            state=root_state,
            model_key=run_spec.model_key,
            seed_refiner_outputs=True,
        )
    except SystemExit as exc:
        _persist_failed_state_context(
            manager=manager,
            run_root=run_root,
            state=root_state,
        )
        return _system_exit_code(exc=exc)
    root_state.save()
    _write_scoreboard(run_root=run_root, states=manager.states)

    next_iteration = max((state.iteration for state in manager.states), default=0) + 1
    while next_iteration <= run_spec.iterations:
        states = manager.states
        best_completed_state = _select_best_completed_state(states=states)
        scoreboard_text = _build_scoreboard(states=states)
        planner_notes_path = _prepare_planner_notes(run_root=run_root, states=states)
        artifacts_dir = run_root / "artifacts" / f"iteration-{next_iteration:04d}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        planner_prompt_text = _render_planner_prompt(
            run_root=run_root,
            latest_state=states[-1] if states else None,
            best_completed_state=best_completed_state,
            scoreboard_text=scoreboard_text,
            benchmark_model_key=run_spec.model_key,
            candidate_state_count=len(manager.completed_states),
            iteration_count=len(states),
        )
        planner_prompt_path = artifacts_dir / "planner_prompt.txt"
        planner_prompt_path.write_text(planner_prompt_text, encoding="utf-8")

        with manager.planning_environment(
            planner_notes_path=planner_notes_path,
        ) as (planning_workspace, candidate_states):
            _copy_planning_inputs(
                planning_workspace=planning_workspace,
                artifacts_dir=artifacts_dir,
            )
            planner_completed = run_planner_agent(
                workspace_path=planning_workspace,
                prompt_text=planner_prompt_text,
                cursor_model=run_spec.cursor_model,
            )
            record_completed_process(
                output_path=artifacts_dir / "planner_step.json",
                completed=planner_completed,
            )
            if planner_completed.returncode != 0:
                _sync_planner_notes(
                    run_root=run_root,
                    planning_workspace=planning_workspace,
                    artifacts_dir=artifacts_dir,
                )
                print(planner_completed.stdout)
                print(planner_completed.stderr, file=sys.stderr)
                return planner_completed.returncode

            planner_output_source = planning_workspace / "output.json"
            planner_output_path = artifacts_dir / "planner_output.json"
            shutil.copy2(src=planner_output_source, dst=planner_output_path)
            _sync_planner_notes(
                run_root=run_root,
                planning_workspace=planning_workspace,
                artifacts_dir=artifacts_dir,
            )
            planning_output = PlanningOutput.load(path=planner_output_path)

            if not 0 <= planning_output.selected_state_index < len(candidate_states):
                raise ValueError(
                    f"Planner selected an out-of-range state index: {planning_output.selected_state_index}"
                )
            parent_state = candidate_states[planning_output.selected_state_index]

        state = manager.create_child_state(
            parent_state=parent_state,
            iteration=next_iteration,
            plan=planning_output.plan,
            planner_selected_state_index=planning_output.selected_state_index,
            planner_prompt_artifact_path=planner_prompt_path,
            planner_output_artifact_path=planner_output_path,
        )
        implementation_prompt_text = _render_implementation_prompt(
            parent_state=parent_state,
            plan=planning_output.plan,
        )
        implementation_prompt_path = artifacts_dir / "implementation_prompt.txt"
        implementation_prompt_path.write_text(
            implementation_prompt_text,
            encoding="utf-8",
        )
        implementation_completed = run_implementation_agent(
            workspace_path=Path(state.refiner_workspace_path),
            prompt_text=implementation_prompt_text,
            cursor_model=run_spec.cursor_model,
        )
        record_completed_process(
            output_path=artifacts_dir / "implementation_step.json",
            completed=implementation_completed,
        )
        if implementation_completed.returncode != 0:
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            print(implementation_completed.stdout)
            print(implementation_completed.stderr, file=sys.stderr)
            return implementation_completed.returncode

        validation_completed = run_workspace_validation(
            workspace_path=Path(state.refiner_workspace_path),
            model_key=run_spec.model_key,
        )
        record_completed_process(
            output_path=artifacts_dir / "validation_step.json",
            completed=validation_completed,
        )
        if validation_completed.returncode != 0:
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            print(validation_completed.stdout)
            print(validation_completed.stderr, file=sys.stderr)
            return validation_completed.returncode

        try:
            _ensure_state_evaluated(
                manager=manager,
                run_root=run_root,
                state=state,
                model_key=run_spec.model_key,
                seed_refiner_outputs=False,
            )
        except SystemExit as exc:
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            return _system_exit_code(exc=exc)
        state.save()
        _write_scoreboard(run_root=run_root, states=manager.states)
        next_iteration += 1

    _write_scoreboard(run_root=run_root, states=manager.states)
    return 0


def _create_run_root(*, outputs_root: Path, run_spec: RunSpec) -> Path:
    timestamp = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    candidate = outputs_root / f"{run_spec.name}-{timestamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = outputs_root / f"{run_spec.name}-{timestamp}-{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_manifest(*, run_root: Path, run_spec: RunSpec) -> None:
    payload = {
        "name": run_spec.name,
        "baseline": run_spec.baseline,
        "model_key": run_spec.model_key,
        "cursor_model": run_spec.cursor_model,
        "iterations": run_spec.iterations,
        "random_seed": run_spec.random_seed,
        "benchmark_tasks": list(run_spec.benchmark_tasks),
    }
    (run_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _ensure_state_evaluated(
    *,
    manager: StateManager,
    run_root: Path,
    state: AgentState,
    model_key: str,
    seed_refiner_outputs: bool,
) -> None:
    if state.result is None or state.official_benchmark is None:
        _benchmark_state(
            run_root=run_root,
            state=state,
            model_key=model_key,
        )
    if seed_refiner_outputs:
        manager.seed_refiner_outputs(state=state)


def _benchmark_state(
    *,
    run_root: Path,
    state: AgentState,
    model_key: str,
) -> None:
    iteration_artifacts = run_root / "artifacts" / f"iteration-{state.iteration:04d}"
    iteration_artifacts.mkdir(parents=True, exist_ok=True)
    benchmark_artifacts_dir = iteration_artifacts / "official_benchmark"
    benchmark_result_path = iteration_artifacts / "benchmark_result.json"
    benchmark_completed = run_workspace_benchmark(
        workspace_path=Path(state.refiner_workspace_path),
        model_key=model_key,
        result_json_out=benchmark_result_path,
        record_visible=False,
        request_label="official",
        artifacts_dir=benchmark_artifacts_dir,
    )
    record_completed_process(
        output_path=iteration_artifacts / "benchmark_step.json",
        completed=benchmark_completed,
    )
    if benchmark_completed.returncode != 0:
        print(benchmark_completed.stdout)
        print(benchmark_completed.stderr, file=sys.stderr)
        raise SystemExit(benchmark_completed.returncode)
    official_benchmark, benchmark_summary = load_workspace_benchmark_result(
        result_json_path=benchmark_result_path,
    )
    state.official_benchmark = official_benchmark
    state.result = benchmark_summary


def _build_scoreboard(*, states: list[AgentState]) -> str:
    lines = [
        "## Scoreboard",
        "",
        "| Iter | Baseline | Reward | Passed | Failed | Errors | Parent |",
        "|------|----------|--------|--------|--------|--------|--------|",
    ]
    for state in states:
        parent = "root"
        if state.prev_path:
            parent = Path(state.prev_path).stem.replace("iteration-", "")
        reward = (
            f"{state.result.reward_mean:.3f}"
            if state.result and state.result.reward_mean is not None
            else "N/A"
        )
        passed = state.result.pass_count if state.result else "N/A"
        failed = state.result.failure_count if state.result else "N/A"
        errors = state.result.error_count if state.result else "N/A"
        lines.append(
            f"| {state.iteration} | {state.baseline} | {reward} | {passed} | {failed} | {errors} | {parent} |"
        )
    return "\n".join(lines)


def _select_best_completed_state(*, states: list[AgentState]) -> AgentState | None:
    completed = [
        state
        for state in states
        if state.result and state.result.reward_mean is not None
    ]
    if not completed:
        return None

    def _sort_key(state: AgentState) -> tuple[float, int, int, int, int]:
        assert state.result is not None
        assert state.result.reward_mean is not None
        return (
            state.result.reward_mean,
            state.result.pass_count,
            -state.result.error_count,
            -state.result.failure_count,
            -state.iteration,
        )

    return max(
        completed,
        key=_sort_key,
    )


def _write_scoreboard(*, run_root: Path, states: list[AgentState]) -> None:
    (run_root / "SCOREBOARD.md").write_text(
        _build_scoreboard(states=states) + "\n",
        encoding="utf-8",
    )


def _render_planner_prompt(
    *,
    run_root: Path,
    latest_state: AgentState | None,
    best_completed_state: AgentState | None,
    scoreboard_text: str,
    benchmark_model_key: str,
    candidate_state_count: int,
    iteration_count: int,
) -> str:
    template = load_planning_prompt()
    latest_result = latest_state.result if latest_state else None
    latest_artifacts = latest_run_artifact_map(
        state=latest_state,
        run_root=run_root,
    )
    best_state = best_completed_state
    best_result = best_state.result if best_state else None
    return template.format(
        run_root=run_root,
        candidate_state_count=candidate_state_count,
        iteration_count=iteration_count,
        benchmark_model_key=benchmark_model_key,
        latest_iteration=latest_state.iteration if latest_state else "N/A",
        latest_status=(
            classify_state_status(state=latest_state, run_root=run_root)
            if latest_state
            else "N/A"
        ),
        latest_reward=(
            f"{latest_result.reward_mean:.3f}"
            if latest_result and latest_result.reward_mean is not None
            else "N/A"
        ),
        latest_passed=latest_result.pass_count if latest_result else "N/A",
        latest_failed=latest_result.failure_count if latest_result else "N/A",
        latest_errors=latest_result.error_count if latest_result else "N/A",
        latest_problem_tasks=(
            summarize_problem_trials(state=latest_state) if latest_state else "N/A"
        ),
        latest_plan_summary=(
            plan_summary(plan=latest_state.plan)
            if latest_state
            else "No plan recorded."
        ),
        latest_selectable=latest_run_selectable_text(state=latest_state),
        latest_parent_iteration=latest_run_parent_iteration_text(state=latest_state),
        latest_benchmark_summary_path=latest_artifacts["benchmark_summary_path"],
        latest_benchmark_stdout_path=latest_artifacts["benchmark_stdout_path"],
        latest_benchmark_stderr_path=latest_artifacts["benchmark_stderr_path"],
        latest_harbor_job_dir=latest_artifacts["harbor_job_dir"],
        latest_implementation_step_path=latest_artifacts["implementation_step_path"],
        latest_validation_step_path=latest_artifacts["validation_step_path"],
        latest_benchmark_step_path=latest_artifacts["benchmark_step_path"],
        latest_benchmark_result_path=latest_artifacts["benchmark_result_path"],
        best_iteration=best_state.iteration if best_state else "N/A",
        best_reward=(
            f"{best_result.reward_mean:.3f}"
            if best_result and best_result.reward_mean is not None
            else "N/A"
        ),
        best_passed=best_result.pass_count if best_result else "N/A",
        best_failed=best_result.failure_count if best_result else "N/A",
        best_errors=best_result.error_count if best_result else "N/A",
        scoreboard=scoreboard_text,
    )


def _render_implementation_prompt(
    *,
    parent_state: AgentState,
    plan: str,
) -> str:
    template = load_implementation_prompt()
    parent_result = parent_state.result
    parent_benchmark = parent_state.official_benchmark
    return template.format(
        parent_reward=(
            f"{parent_result.reward_mean:.3f}"
            if parent_result and parent_result.reward_mean is not None
            else "N/A"
        ),
        parent_passed=parent_result.pass_count if parent_result else "N/A",
        parent_failed=parent_result.failure_count if parent_result else "N/A",
        parent_errors=parent_result.error_count if parent_result else "N/A",
        parent_benchmark_summary_path=(
            parent_benchmark.benchmark_summary_path if parent_benchmark else "N/A"
        ),
        parent_benchmark_stdout_path=(
            parent_benchmark.benchmark_stdout_path if parent_benchmark else "N/A"
        ),
        parent_benchmark_stderr_path=(
            parent_benchmark.benchmark_stderr_path if parent_benchmark else "N/A"
        ),
        parent_harbor_job_dir=(
            parent_benchmark.harbor_job_dir if parent_benchmark else "N/A"
        ),
        plan=plan,
    )


def _copy_planning_inputs(*, planning_workspace: Path, artifacts_dir: Path) -> None:
    copies = {
        planning_workspace / "states.json": artifacts_dir / "planner_states.json",
        planning_workspace / "state-schema.json": artifacts_dir
        / "planner_state_schema.json",
        planning_workspace / "output-schema.json": artifacts_dir
        / "planner_output_schema.json",
        planning_workspace / PLANNER_NOTES_FILE_NAME: artifacts_dir
        / "planner_notes_input.md",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src=src, dst=dst)


def _planner_notes_path(*, run_root: Path) -> Path:
    return run_root / PLANNER_NOTES_FILE_NAME


def _prepare_planner_notes(*, run_root: Path, states: list[AgentState]) -> Path:
    notes_path = _planner_notes_path(run_root=run_root)
    if notes_path.exists():
        content = notes_path.read_text(encoding="utf-8")
    else:
        content = planner_notes_template()
    latest_state = states[-1] if states else None
    if latest_state is not None:
        iteration_header = latest_iteration_header(state=latest_state)
        if iteration_header not in content:
            content = (
                content.rstrip()
                + "\n\n"
                + latest_iteration_section(
                    state=latest_state,
                    run_root=run_root,
                )
            )
    notes_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return notes_path


def _sync_planner_notes(
    *, run_root: Path, planning_workspace: Path, artifacts_dir: Path
) -> None:
    notes_path = planning_workspace / PLANNER_NOTES_FILE_NAME
    if not notes_path.exists():
        return
    shutil.copy2(src=notes_path, dst=_planner_notes_path(run_root=run_root))
    shutil.copy2(src=notes_path, dst=artifacts_dir / "planner_notes_output.md")


def _persist_failed_state_context(
    *, manager: StateManager, run_root: Path, state: AgentState
) -> None:
    state.save()
    _write_scoreboard(run_root=run_root, states=manager.states)


def _system_exit_code(*, exc: SystemExit) -> int:
    return exc.code if isinstance(exc.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
