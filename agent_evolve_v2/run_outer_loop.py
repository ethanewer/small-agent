# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import sys

from agent_evolve_v2.benchmark import (
    load_workspace_benchmark_result,
    run_workspace_benchmark,
)
from agent_evolve_v2.config import RunSpec, load_runs_config
from agent_evolve_v2.runtime import (
    record_completed_process,
    run_cursor_refiner,
    run_workspace_critique,
    run_workspace_validation,
)
from agent_evolve_v2.state import AgentState, CriticItem, CriticSummary
from agent_evolve_v2.state_manager import ParentSample, StateManager
from agent_evolve_v2.workspace_support.benchmark_cache import (
    resolve_latest_visible_run_dir,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-contained agent_evolve_v2 loop.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent_evolve_v2/runs.json"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "agent_evolve_v2" / "outputs"
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
            max_critic_failures=int(manifest["max_critic_failures"]),
            max_critic_successes=int(manifest["max_critic_successes"]),
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
                max_critic_failures=run_spec.max_critic_failures,
                max_critic_successes=run_spec.max_critic_successes,
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
    rng = random.Random(run_spec.random_seed)

    root_state = manager.bootstrap_root_state()
    _ensure_state_evaluated_and_critiqued(
        manager=manager,
        run_root=run_root,
        state=root_state,
        model_key=run_spec.model_key,
        seed_refiner_outputs=True,
        max_critic_failures=run_spec.max_critic_failures,
        max_critic_successes=run_spec.max_critic_successes,
    )
    root_state.save()

    next_iteration = max((state.iteration for state in manager.states), default=0) + 1
    while next_iteration <= run_spec.iterations:
        parent_sample = manager.sample_parent_state(rng=rng)
        scoreboard_text = _build_scoreboard(states=manager.states)
        best_completed_state = _select_best_completed_state(states=manager.states)
        state = manager.create_child_state(
            parent_state=parent_sample.state,
            iteration=next_iteration,
            plan=(
                "Refine the selected parent workspace using the critic summary "
                f"from iteration {parent_sample.state.iteration}."
            ),
        )
        prompt_text = _render_prompt(
            run_root=run_root,
            workspace_root=Path(state.refiner_workspace_path),
            parent_sample=parent_sample,
            best_completed_state=best_completed_state,
            scoreboard_text=scoreboard_text,
            benchmark_model_key=run_spec.model_key,
        )
        artifacts_dir = run_root / "artifacts" / f"iteration-{next_iteration:04d}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = artifacts_dir / "cursor_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        _seed_workspace_notes(
            state=state,
            parent_sample=parent_sample,
            best_completed_state=best_completed_state,
            scoreboard_text=scoreboard_text,
        )
        cursor_completed = run_cursor_refiner(
            workspace_path=Path(state.refiner_workspace_path),
            prompt_text=prompt_text,
            cursor_model=run_spec.cursor_model,
        )
        record_completed_process(
            output_path=artifacts_dir / "cursor_step.json",
            completed=cursor_completed,
        )
        if cursor_completed.returncode != 0:
            print(cursor_completed.stdout)
            print(cursor_completed.stderr, file=sys.stderr)
            return cursor_completed.returncode

        validation_completed = run_workspace_validation(
            workspace_path=Path(state.refiner_workspace_path),
            model_key=run_spec.model_key,
        )
        record_completed_process(
            output_path=artifacts_dir / "validation_step.json",
            completed=validation_completed,
        )
        if validation_completed.returncode != 0:
            print(validation_completed.stdout)
            print(validation_completed.stderr, file=sys.stderr)
            return validation_completed.returncode

        _ensure_state_evaluated_and_critiqued(
            manager=manager,
            run_root=run_root,
            state=state,
            model_key=run_spec.model_key,
            seed_refiner_outputs=False,
            max_critic_failures=run_spec.max_critic_failures,
            max_critic_successes=run_spec.max_critic_successes,
        )
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
        "max_critic_failures": run_spec.max_critic_failures,
        "max_critic_successes": run_spec.max_critic_successes,
        "random_seed": run_spec.random_seed,
        "benchmark_tasks": list(run_spec.benchmark_tasks),
    }
    (run_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _ensure_state_evaluated_and_critiqued(
    *,
    manager: StateManager,
    run_root: Path,
    state: AgentState,
    model_key: str,
    seed_refiner_outputs: bool,
    max_critic_failures: int,
    max_critic_successes: int,
) -> None:
    if state.result is None or state.official_benchmark is None:
        _benchmark_state(
            run_root=run_root,
            state=state,
            model_key=model_key,
        )
    if seed_refiner_outputs:
        manager.seed_refiner_outputs(state=state)
    if state.critic is None or state.critic_workspace_path is None:
        _criticize_state(
            manager=manager,
            run_root=run_root,
            state=state,
            max_critic_failures=max_critic_failures,
            max_critic_successes=max_critic_successes,
        )


def _benchmark_state(
    *,
    run_root: Path,
    state: AgentState,
    model_key: str,
) -> None:
    iteration_artifacts = run_root / "artifacts" / f"iteration-{state.iteration:04d}"
    iteration_artifacts.mkdir(parents=True, exist_ok=True)
    benchmark_result_path = iteration_artifacts / "benchmark_result.json"
    benchmark_completed = run_workspace_benchmark(
        workspace_path=Path(state.refiner_workspace_path),
        model_key=model_key,
        result_json_out=benchmark_result_path,
        record_visible=False,
        request_label="official",
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


def _criticize_state(
    *,
    manager: StateManager,
    run_root: Path,
    state: AgentState,
    max_critic_failures: int,
    max_critic_successes: int,
) -> None:
    critic_workspace = manager.create_critic_workspace(state=state)
    critique_completed = run_workspace_critique(
        workspace_path=critic_workspace,
        max_failures=max_critic_failures,
        max_successes=max_critic_successes,
    )
    iteration_artifacts = run_root / "artifacts" / f"iteration-{state.iteration:04d}"
    record_completed_process(
        output_path=iteration_artifacts / "critic_step.json",
        completed=critique_completed,
    )
    if critique_completed.returncode != 0:
        print(critique_completed.stdout)
        print(critique_completed.stderr, file=sys.stderr)
        raise SystemExit(critique_completed.returncode)
    latest_visible_run_dir = resolve_latest_visible_run_dir(
        workspace_root=critic_workspace,
    )
    critic_payload = json.loads(
        (latest_visible_run_dir / "critic_summary.json").read_text(encoding="utf-8")
    )
    state.critic = _load_critic_summary(payload=critic_payload)
    state.notes = state.critic.summary_markdown


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


def _render_prompt(
    *,
    run_root: Path,
    workspace_root: Path,
    parent_sample: ParentSample,
    best_completed_state: AgentState | None,
    scoreboard_text: str,
    benchmark_model_key: str,
) -> str:
    template = (Path(__file__).with_name("headless_inner_loop_prompt.md")).read_text(
        encoding="utf-8"
    )
    best_state = best_completed_state or parent_sample.state
    best_result = best_state.result
    critic_summary = (
        parent_sample.state.critic.summary_markdown
        if parent_sample.state.critic
        else "No critic summary available."
    )
    return template.format(
        run_root=run_root,
        workspace_root=workspace_root,
        parent_iteration=parent_sample.state.iteration,
        parent_workspace=parent_sample.state.refiner_workspace_path,
        baseline=parent_sample.state.baseline,
        thompson_sample=f"{parent_sample.score_sample:.6f}",
        alpha=f"{parent_sample.alpha:.2f}",
        beta=f"{parent_sample.beta:.2f}",
        benchmark_model_key=benchmark_model_key,
        best_iteration=best_state.iteration,
        best_reward=(
            f"{best_result.reward_mean:.3f}"
            if best_result and best_result.reward_mean is not None
            else "N/A"
        ),
        best_passed=best_result.pass_count if best_result else "N/A",
        best_failed=best_result.failure_count if best_result else "N/A",
        best_errors=best_result.error_count if best_result else "N/A",
        best_workspace=best_state.refiner_workspace_path,
        scoreboard=scoreboard_text,
        critic_summary=critic_summary,
    )


def _seed_workspace_notes(
    *,
    state: AgentState,
    parent_sample: ParentSample,
    best_completed_state: AgentState | None,
    scoreboard_text: str,
) -> None:
    notes_path = Path(state.refiner_workspace_path) / "NOTES.md"
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    best_state = best_completed_state or parent_sample.state
    best_result = best_state.result
    auto_block = "\n".join(
        [
            "<!-- BEGIN AUTO STATE CONTEXT -->",
            f"Selected parent iteration: {parent_sample.state.iteration}",
            f"Baseline: {parent_sample.state.baseline}",
            f"Thompson sample: {parent_sample.score_sample:.6f}",
            f"Posterior alpha/beta: {parent_sample.alpha:.2f}/{parent_sample.beta:.2f}",
            f"Best completed iteration so far: {best_state.iteration}",
            (
                "Best completed reward / passed / failed / errors: "
                + (
                    " / ".join(
                        [
                            f"{best_result.reward_mean:.3f}",
                            str(best_result.pass_count),
                            str(best_result.failure_count),
                            str(best_result.error_count),
                        ]
                    )
                    if best_result and best_result.reward_mean is not None
                    else "N/A"
                )
            ),
            f"Best workspace (reference only): {best_state.refiner_workspace_path}",
            "",
            scoreboard_text,
            "",
            "## Parent critic summary",
            "",
            parent_sample.state.critic.summary_markdown
            if parent_sample.state.critic
            else "No critic summary available.",
            "<!-- END AUTO STATE CONTEXT -->",
            "",
        ]
    )
    stripped_existing = existing
    if "<!-- BEGIN AUTO STATE CONTEXT -->" in existing:
        _, _, remainder = existing.partition("<!-- BEGIN AUTO STATE CONTEXT -->")
        _, _, stripped_existing = remainder.partition("<!-- END AUTO STATE CONTEXT -->")
        stripped_existing = stripped_existing.lstrip("\n")
    notes_path.write_text(auto_block + stripped_existing, encoding="utf-8")


def _load_critic_summary(*, payload: dict[str, object]) -> CriticSummary:
    items = []
    raw_items = payload.get("items", [])
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                items.append(CriticItem(**item))
    return CriticSummary(
        created_at_utc=str(payload.get("created_at_utc", "")),
        source_job_dir=str(payload.get("source_job_dir", "")),
        summary_markdown=str(payload.get("summary_markdown", "")),
        items=items,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
