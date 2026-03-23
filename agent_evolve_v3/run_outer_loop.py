# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import time

from agent_evolve_v3.config import RunSpec, load_runs_config
from agent_evolve_v3.prompts import (
    load_failure_investigation_prompt,
    load_implementation_prompt,
    load_planning_prompt,
)
from agent_evolve_v3.services.benchmark import (
    merge_sample_results,
    prepull_task_images,
    run_n_sample_benchmarks,
)
from agent_evolve_v3.services.runtime import (
    record_completed_process,
    run_failure_investigation_agent,
    run_implementation_agent,
    run_planner_agent,
    run_workspace_validation,
)
from agent_evolve_v3.state import AgentState, PlanningOutput
from agent_evolve_v3.state.types import BenchmarkSummary, FailureAnalysis
from agent_evolve_v3.state.manager import StateManager
from agent_evolve_v3.state.planner_context import (
    PLANNER_NOTES_FILE_NAME,
    build_task_pass_rate_table,
    classify_state_status,
    compute_noise_stats,
    format_failure_analyses,
    latest_iteration_header,
    latest_iteration_section,
    latest_run_artifact_map,
    latest_run_parent_iteration_text,
    latest_run_selectable_text,
    plan_summary,
    planner_notes_template,
    summarize_problem_trials,
    summarize_problem_trials_detail,
)


def _log(msg: str) -> None:
    timestamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def _format_elapsed(start: float) -> str:
    elapsed = time.monotonic() - start
    minutes, seconds = divmod(int(elapsed), 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


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
            train_small_tasks=tuple(
                str(task_name) for task_name in manifest.get("train_small_tasks", [])
            ),
            n_samples=int(manifest.get("n_samples", 2)),
            failure_investigation_model=str(
                manifest.get("failure_investigation_model", "gemini-3-flash")
            ),
        )
        _log(f"Resuming run '{run_spec.name}' from {run_root.name}")
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
                train_small_tasks=run_spec.train_small_tasks,
                n_samples=run_spec.n_samples,
                failure_investigation_model=run_spec.failure_investigation_model,
            )
        run_root = _create_run_root(outputs_root=outputs_root, run_spec=run_spec)
        _log(f"Starting new run '{run_spec.name}' at {run_root.name}")

    _log(
        f"Config: model={run_spec.model_key}, cursor={run_spec.cursor_model}, "
        f"iterations={run_spec.iterations}, tasks={len(run_spec.benchmark_tasks)}, "
        f"train_small={len(run_spec.train_small_tasks)}, n_samples={run_spec.n_samples}, "
        f"failure_inv_model={run_spec.failure_investigation_model}"
    )

    manager = StateManager(
        repo_root=repo_root,
        run_root=run_root,
        run_spec=run_spec,
    )
    _write_manifest(run_root=run_root, run_spec=run_spec)

    _log("Pre-pulling Docker images...")
    t0 = time.monotonic()
    prepull_task_images(
        task_names=run_spec.benchmark_tasks,
        repo_root=repo_root,
    )
    _log(f"Pre-pull finished ({_format_elapsed(t0)})")

    root_state = manager.bootstrap_root_state()
    if root_state.result is None:
        _log("Iteration 0 (root): running benchmark...")
    else:
        _log(
            f"Iteration 0 (root): already evaluated "
            f"(reward={root_state.result.reward_mean})"
        )

    try:
        _ensure_state_evaluated(
            manager=manager,
            run_root=run_root,
            state=root_state,
            run_spec=run_spec,
            seed_refiner_outputs=True,
        )
    except SystemExit as exc:
        _persist_failed_state_context(
            manager=manager,
            run_root=run_root,
            state=root_state,
        )
        return _system_exit_code(exc=exc)

    root_result = root_state.result
    if root_result and root_result.sample_results and not root_state.failure_analyses:
        _log("Running failure investigations for root state...")
        t0 = time.monotonic()
        root_state.failure_analyses = _run_failure_investigations(
            state=root_state,
            run_spec=run_spec,
            sample_summaries=root_result.sample_results,
        )
        _log(
            f"Failure investigation done ({_format_elapsed(t0)}): "
            f"{len(root_state.failure_analyses)} reports"
        )

    root_state.save()
    _write_scoreboard(run_root=run_root, states=manager.states)

    existing_states = manager.states
    _log(f"Loaded {len(existing_states)} existing state(s)")

    next_iteration = max((state.iteration for state in existing_states), default=0) + 1
    _log(f"Starting from iteration {next_iteration}/{run_spec.iterations}")

    while next_iteration <= run_spec.iterations:
        iter_start = time.monotonic()
        _log(f"{'=' * 60}")
        _log(f"Iteration {next_iteration}/{run_spec.iterations}")
        _log(f"{'=' * 60}")

        states = manager.states
        best_completed_state = _select_best_completed_state(states=states)
        if best_completed_state and best_completed_state.result:
            _log(
                f"Best so far: iteration {best_completed_state.iteration} "
                f"(reward={best_completed_state.result.reward_mean})"
            )

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
            completed_states=manager.completed_states,
        )
        planner_prompt_path = artifacts_dir / "planner_prompt.txt"
        planner_prompt_path.write_text(planner_prompt_text, encoding="utf-8")

        _log("Running planner agent...")
        t0 = time.monotonic()
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
                _log(
                    f"Planner FAILED (rc={planner_completed.returncode}, {_format_elapsed(t0)})"
                )
                _sync_planner_notes(
                    run_root=run_root,
                    planning_workspace=planning_workspace,
                    artifacts_dir=artifacts_dir,
                )
                print(planner_completed.stdout, flush=True)
                print(planner_completed.stderr, file=sys.stderr, flush=True)
                return planner_completed.returncode

            _log(f"Planner completed ({_format_elapsed(t0)})")

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

        _log(
            f"Planner selected parent iteration {parent_state.iteration} "
            f"(index {planning_output.selected_state_index})"
        )

        state = manager.create_child_state(
            parent_state=parent_state,
            iteration=next_iteration,
            plan=planning_output.plan,
            planner_selected_state_index=planning_output.selected_state_index,
            planner_prompt_artifact_path=planner_prompt_path,
            planner_output_artifact_path=planner_output_path,
        )

        _log("Running implementation agent...")
        t0 = time.monotonic()
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
            _log(
                f"Implementation FAILED (rc={implementation_completed.returncode}, {_format_elapsed(t0)})"
            )
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            print(implementation_completed.stdout, flush=True)
            print(implementation_completed.stderr, file=sys.stderr, flush=True)
            return implementation_completed.returncode

        _log(f"Implementation completed ({_format_elapsed(t0)})")

        _log("Running validation...")
        t0 = time.monotonic()
        validation_completed = run_workspace_validation(
            workspace_path=Path(state.refiner_workspace_path),
            model_key=run_spec.model_key,
        )
        record_completed_process(
            output_path=artifacts_dir / "validation_step.json",
            completed=validation_completed,
        )
        if validation_completed.returncode != 0:
            _log(
                f"Validation FAILED (rc={validation_completed.returncode}, {_format_elapsed(t0)})"
            )
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            print(validation_completed.stdout, flush=True)
            print(validation_completed.stderr, file=sys.stderr, flush=True)
            return validation_completed.returncode

        _log(f"Validation passed ({_format_elapsed(t0)})")

        _log("Running benchmark...")
        t0 = time.monotonic()
        try:
            _ensure_state_evaluated(
                manager=manager,
                run_root=run_root,
                state=state,
                run_spec=run_spec,
                seed_refiner_outputs=False,
            )
        except SystemExit as exc:
            _log(f"Benchmark FAILED ({_format_elapsed(t0)})")
            _persist_failed_state_context(
                manager=manager,
                run_root=run_root,
                state=state,
            )
            return _system_exit_code(exc=exc)

        result = state.result
        reward = result.reward_mean if result else None
        passed = result.pass_count if result else 0
        failed = result.failure_count if result else 0
        errors = result.error_count if result else 0
        _log(
            f"Benchmark done ({_format_elapsed(t0)}): "
            f"reward={reward}, passed={passed}, failed={failed}, errors={errors}"
        )

        if result and result.sample_results:
            _log("Running failure investigations...")
            t0 = time.monotonic()
            state.failure_analyses = _run_failure_investigations(
                state=state,
                run_spec=run_spec,
                sample_summaries=result.sample_results,
            )
            _log(
                f"Failure investigation done ({_format_elapsed(t0)}): "
                f"{len(state.failure_analyses)} reports"
            )

        state.save()
        _write_scoreboard(run_root=run_root, states=manager.states)
        _log(f"Iteration {next_iteration} total time: {_format_elapsed(iter_start)}")
        next_iteration += 1

    _write_scoreboard(run_root=run_root, states=manager.states)
    _log("Run complete.")
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
        "train_small_tasks": list(run_spec.train_small_tasks),
        "n_samples": run_spec.n_samples,
        "failure_investigation_model": run_spec.failure_investigation_model,
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
    run_spec: RunSpec,
    seed_refiner_outputs: bool,
) -> None:
    if state.result is None or state.official_benchmark is None:
        _benchmark_state_two_tier(
            run_root=run_root,
            state=state,
            run_spec=run_spec,
            completed_states=manager.completed_states,
        )
    if seed_refiner_outputs:
        manager.seed_refiner_outputs(state=state)


def _benchmark_state_two_tier(
    *,
    run_root: Path,
    state: AgentState,
    run_spec: RunSpec,
    completed_states: list[AgentState],
) -> None:
    iteration_artifacts = run_root / "artifacts" / f"iteration-{state.iteration:04d}"
    iteration_artifacts.mkdir(parents=True, exist_ok=True)

    is_root = state.iteration == 0
    all_tasks = list(run_spec.benchmark_tasks)
    small_tasks = list(run_spec.train_small_tasks)
    remaining_tasks = [t for t in all_tasks if t not in set(small_tasks)]
    n_samples = run_spec.n_samples

    if is_root:
        _log("  Running train/full benchmark (iteration 0 baseline)...")
        _run_full_benchmark(
            state=state,
            iteration_artifacts=iteration_artifacts,
            run_spec=run_spec,
            all_tasks=all_tasks,
            small_tasks=small_tasks,
            n_samples=n_samples,
        )
    else:
        _log(
            f"  Running train/small benchmark ({len(small_tasks)} tasks, N={n_samples})..."
        )
        small_results = run_n_sample_benchmarks(
            workspace_path=Path(state.refiner_workspace_path),
            model_key=run_spec.model_key,
            task_names=small_tasks,
            artifacts_base_dir=iteration_artifacts / "train_small",
            n_samples=n_samples,
        )
        small_summaries = [s for _, s in small_results]
        small_reward, _sample_list = merge_sample_results(
            sample_summaries=small_summaries,
            n_tasks=len(small_tasks),
        )
        _log(f"  train/small reward: {small_reward:.3f}")

        last_official_run = small_results[-1][0] if small_results else None

        best_small_reward = _best_train_small_reward(states=completed_states)
        promoted = small_reward > best_small_reward
        _log(
            f"  Promotion check: {small_reward:.3f} > "
            f"{best_small_reward:.3f} -> "
            f"{'PROMOTED' if promoted else 'not promoted'}"
        )

        if promoted:
            _log(f"  Running remaining {len(remaining_tasks)} tasks (N={n_samples})...")
            remaining_results = run_n_sample_benchmarks(
                workspace_path=Path(state.refiner_workspace_path),
                model_key=run_spec.model_key,
                task_names=remaining_tasks,
                artifacts_base_dir=iteration_artifacts / "train_remaining",
                n_samples=n_samples,
            )
            remaining_summaries = [s for _, s in remaining_results]
            total_small_passes = sum(s.pass_count for s in small_summaries)
            total_remaining_passes = sum(s.pass_count for s in remaining_summaries)
            full_reward = (total_small_passes + total_remaining_passes) / (
                n_samples * len(all_tasks)
            )
            _log(f"  train/full reward: {full_reward:.3f}")

            all_passed = []
            all_failed = []
            for s in small_summaries + remaining_summaries:
                all_passed.extend(s.passed_trials)
                all_failed.extend(s.failed_trials)

            if remaining_results:
                last_official_run = remaining_results[-1][0]

            state.result = BenchmarkSummary(
                created_at_utc=last_official_run.created_at_utc
                if last_official_run
                else "",
                aggregate_result_path=last_official_run.aggregate_result_path
                if last_official_run
                else "",
                harbor_job_dir=last_official_run.harbor_job_dir
                if last_official_run
                else "",
                reward_mean=full_reward,
                n_trials=len(all_tasks) * n_samples,
                pass_count=total_small_passes + total_remaining_passes,
                failure_count=sum(
                    s.failure_count for s in small_summaries + remaining_summaries
                ),
                error_count=sum(
                    s.error_count for s in small_summaries + remaining_summaries
                ),
                passed_trials=all_passed,
                failed_trials=all_failed,
                train_small_reward_mean=small_reward,
                train_small_n_samples=n_samples,
                train_full_reward_mean=full_reward,
                train_full_n_samples=n_samples,
                sample_results=small_summaries + remaining_summaries,
            )
        else:
            all_passed = []
            all_failed = []
            for s in small_summaries:
                all_passed.extend(s.passed_trials)
                all_failed.extend(s.failed_trials)

            state.result = BenchmarkSummary(
                created_at_utc=last_official_run.created_at_utc
                if last_official_run
                else "",
                aggregate_result_path=last_official_run.aggregate_result_path
                if last_official_run
                else "",
                harbor_job_dir=last_official_run.harbor_job_dir
                if last_official_run
                else "",
                reward_mean=small_reward,
                n_trials=len(small_tasks) * n_samples,
                pass_count=sum(s.pass_count for s in small_summaries),
                failure_count=sum(s.failure_count for s in small_summaries),
                error_count=sum(s.error_count for s in small_summaries),
                passed_trials=all_passed,
                failed_trials=all_failed,
                train_small_reward_mean=small_reward,
                train_small_n_samples=n_samples,
                sample_results=small_summaries,
            )

        if last_official_run:
            state.official_benchmark = last_official_run


def _run_full_benchmark(
    *,
    state: AgentState,
    iteration_artifacts: Path,
    run_spec: RunSpec,
    all_tasks: list[str],
    small_tasks: list[str],
    n_samples: int,
) -> None:
    full_results = run_n_sample_benchmarks(
        workspace_path=Path(state.refiner_workspace_path),
        model_key=run_spec.model_key,
        task_names=all_tasks,
        artifacts_base_dir=iteration_artifacts / "train_full",
        n_samples=n_samples,
    )
    full_summaries = [s for _, s in full_results]
    full_reward, _ = merge_sample_results(
        sample_summaries=full_summaries,
        n_tasks=len(all_tasks),
    )

    small_task_set = set(small_tasks)
    small_passes = 0
    for s in full_summaries:
        for trial_id in s.passed_trials:
            task_name = trial_id.split("__")[0]
            if task_name in small_task_set:
                small_passes += 1

    small_reward = small_passes / (n_samples * len(small_tasks)) if small_tasks else 0.0
    _log(
        f"  train/full reward: {full_reward:.3f}, train/small reward: {small_reward:.3f}"
    )

    all_passed = []
    all_failed = []
    for s in full_summaries:
        all_passed.extend(s.passed_trials)
        all_failed.extend(s.failed_trials)

    last_official_run = full_results[-1][0] if full_results else None
    state.result = BenchmarkSummary(
        created_at_utc=last_official_run.created_at_utc if last_official_run else "",
        aggregate_result_path=last_official_run.aggregate_result_path
        if last_official_run
        else "",
        harbor_job_dir=last_official_run.harbor_job_dir if last_official_run else "",
        reward_mean=full_reward,
        n_trials=len(all_tasks) * n_samples,
        pass_count=sum(s.pass_count for s in full_summaries),
        failure_count=sum(s.failure_count for s in full_summaries),
        error_count=sum(s.error_count for s in full_summaries),
        passed_trials=all_passed,
        failed_trials=all_failed,
        train_small_reward_mean=small_reward,
        train_small_n_samples=n_samples,
        train_full_reward_mean=full_reward,
        train_full_n_samples=n_samples,
        sample_results=full_summaries,
    )
    if last_official_run:
        state.official_benchmark = last_official_run


def _best_train_small_reward(*, states: list[AgentState]) -> float:
    best = 0.0
    for state in states:
        result = state.result
        if result is None:
            continue
        if result.train_small_reward_mean is not None:
            best = max(best, result.train_small_reward_mean)
    return best


def _run_failure_investigations(
    *,
    state: AgentState,
    run_spec: RunSpec,
    sample_summaries: list[BenchmarkSummary],
) -> list[FailureAnalysis]:
    if len(sample_summaries) < 2:
        return []

    s0, s1 = sample_summaries[0], sample_summaries[1]

    all_task_names: set[str] = set()
    for s in (s0, s1):
        for trial_id in s.passed_trials + s.failed_trials:
            all_task_names.add(trial_id.split("__")[0])
        for trial_ids in s.exception_types.values():
            for trial_id in trial_ids:
                all_task_names.add(trial_id.split("__")[0])

    reward_by_task: dict[str, list[float]] = {}
    for idx, s in enumerate((s0, s1)):
        passed_tasks = {tid.split("__")[0] for tid in s.passed_trials}
        for task_name in all_task_names:
            reward_by_task.setdefault(task_name, [0.0, 0.0])
            if task_name in passed_tasks:
                reward_by_task[task_name][idx] = 1.0

    log_by_task: dict[str, list[str]] = {}
    verifier_by_task: dict[str, list[str]] = {}
    exception_by_task: dict[str, list[str]] = {}
    for idx, s in enumerate((s0, s1)):
        harbor_dir = Path(s.harbor_job_dir)
        if not harbor_dir.is_dir():
            continue
        for trial_dir in harbor_dir.iterdir():
            if not trial_dir.is_dir() or "__" not in trial_dir.name:
                continue
            task_name = trial_dir.name.split("__")[0]
            log_by_task.setdefault(task_name, ["", ""])
            verifier_by_task.setdefault(task_name, ["", ""])
            exception_by_task.setdefault(task_name, ["", ""])

            result_path = trial_dir / "result.json"
            if result_path.exists():
                try:
                    trial_result = json.loads(result_path.read_text(encoding="utf-8"))
                    agent_result = trial_result.get("agent_result", {})
                    metadata = (
                        agent_result.get("metadata", {})
                        if isinstance(agent_result, dict)
                        else {}
                    )
                    workspace_agent = (
                        metadata.get("workspace_agent", {})
                        if isinstance(metadata, dict)
                        else {}
                    )
                    stdout = (
                        str(workspace_agent.get("stdout", ""))
                        if isinstance(workspace_agent, dict)
                        else ""
                    )
                    if not stdout and isinstance(metadata, dict):
                        stdout = str(metadata.get("stdout", ""))
                    log_by_task[task_name][idx] = (
                        stdout[-10000:] if len(stdout) > 10000 else stdout
                    )
                except (json.JSONDecodeError, OSError):
                    pass

            exception_path = trial_dir / "exception.txt"
            if exception_path.exists():
                try:
                    exc_content = exception_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    exception_by_task[task_name][idx] = (
                        exc_content[-3000:] if len(exc_content) > 3000 else exc_content
                    )
                except OSError:
                    pass

            verifier_path = trial_dir / "verifier" / "test-stdout.txt"
            if verifier_path.exists():
                try:
                    content = verifier_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    verifier_by_task[task_name][idx] = (
                        content[-5000:] if len(content) > 5000 else content
                    )
                except OSError:
                    pass

    core_agent_path = Path(state.refiner_workspace_path) / "agent" / "core_agent.py"
    core_agent_source = ""
    if core_agent_path.exists():
        try:
            core_agent_source = core_agent_path.read_text(encoding="utf-8")
        except OSError:
            pass

    investigation_template = load_failure_investigation_prompt()
    analyses: list[FailureAnalysis] = []

    for task_name in sorted(all_task_names):
        rewards = reward_by_task.get(task_name, [0.0, 0.0])
        if rewards[0] == 1.0 and rewards[1] == 1.0:
            continue

        _log(f"    Investigating: {task_name}")
        logs = log_by_task.get(task_name, ["", ""])
        verifiers = verifier_by_task.get(task_name, ["", ""])
        exceptions = exception_by_task.get(task_name, ["", ""])

        prompt_text = investigation_template.format(
            task_name=task_name,
            run_0_reward=rewards[0],
            run_1_reward=rewards[1],
        )

        result = run_failure_investigation_agent(
            prompt_text=prompt_text,
            cursor_model=run_spec.failure_investigation_model,
            run_0_agent_log=logs[0],
            run_1_agent_log=logs[1],
            run_0_verifier=verifiers[0],
            run_1_verifier=verifiers[1],
            run_0_exception=exceptions[0],
            run_1_exception=exceptions[1],
            core_agent_source=core_agent_source,
        )

        analyses.append(
            FailureAnalysis(
                task_name=result.get("task_name", task_name),
                general_failure_reason=result.get(
                    "general_failure_reason", "infrastructure_error"
                ),
                task_specific_explanation=result.get("task_specific_explanation", ""),
                consistency=result.get("consistency", "both_same_failure"),
                suggested_fix_category=result.get(
                    "suggested_fix_category", "not_fixable_by_agent"
                ),
            )
        )

    return analyses


def _build_scoreboard(*, states: list[AgentState]) -> str:
    lines = [
        "## Scoreboard",
        "",
        "| Iter | Baseline | Small (N=2) | Full (N=2) | Passed | Failed | Errors | Parent |",
        "|------|----------|-------------|------------|--------|--------|--------|--------|",
    ]
    for state in states:
        parent = "root"
        if state.prev_path:
            parent = Path(state.prev_path).stem.replace("iteration-", "")
        small_reward = (
            f"{state.result.train_small_reward_mean:.3f}"
            if state.result and state.result.train_small_reward_mean is not None
            else "N/A"
        )
        full_reward = (
            f"{state.result.train_full_reward_mean:.3f}"
            if state.result and state.result.train_full_reward_mean is not None
            else "N/A"
        )
        passed = state.result.pass_count if state.result else "N/A"
        failed = state.result.failure_count if state.result else "N/A"
        errors = state.result.error_count if state.result else "N/A"
        lines.append(
            f"| {state.iteration} | {state.baseline} | {small_reward} | {full_reward} | {passed} | {failed} | {errors} | {parent} |"
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

    def _sort_key(state: AgentState) -> tuple[float, float, int, int, int, int]:
        assert state.result is not None
        assert state.result.reward_mean is not None
        full_reward = state.result.train_full_reward_mean or 0.0
        small_reward = state.result.train_small_reward_mean or state.result.reward_mean
        return (
            full_reward,
            small_reward,
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
    completed_states: list[AgentState],
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
        latest_trial_summaries_path=latest_artifacts["trial_summaries_path"],
        latest_trial_logs_dir=latest_artifacts["trial_logs_dir"],
        latest_problem_trial_details=summarize_problem_trials_detail(
            state=latest_state,
        ),
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
        noise_context=compute_noise_stats(states=completed_states),
        task_pass_rate_table=build_task_pass_rate_table(states=completed_states),
        failure_analysis_summary=(
            format_failure_analyses(state=latest_state)
            if latest_state
            else "No failure analyses available."
        ),
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
