# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from agent_evolve_v3.state import AgentState, BenchmarkSummary, OfficialBenchmarkRun

PLANNER_CONTEXT_FILE_NAME = "planning_context.json"
PLAN_HISTORY_FILE_NAME = "plan_history.md"
FAILED_TASK_INDEX_FILE_NAME = "failed_task_index.json"
FAILED_TASK_TRAJECTORIES_FILE_NAME = "failed_task_trajectories.md"
SCORE_ANALYSIS_FILE_NAME = "score_analysis.md"
PLANNER_NOTES_FILE_NAME = "PLANNER_NOTES.md"

_TRIAL_ARTIFACT_RELATIVE_PATHS = (
    Path("result.json"),
    Path("agent") / "trajectory.json",
    Path("agent") / "transcript.json",
    Path("agent") / "history.json",
    Path("agent") / "stdout.log",
    Path("agent") / "stderr.log",
    Path("verifier") / "test-stdout.txt",
    Path("verifier") / "test-stderr.txt",
)


@dataclass(frozen=True)
class PlannerContextBundle:
    planning_context_payload: dict[str, Any]
    plan_history_text: str
    failed_task_index_payload: dict[str, Any]
    failed_task_trajectories_text: str
    score_analysis_text: str


def build_planner_context_bundle(
    *, states: list[AgentState], run_root: Path
) -> PlannerContextBundle:
    completed_states = [state for state in states if state.result is not None]
    best_completed_state = _best_completed_state(states=states)
    iteration_entries: list[dict[str, Any]] = []
    failed_iterations: list[dict[str, Any]] = []
    delta_entries: list[dict[str, Any]] = []
    state_by_iteration = {state.iteration: state for state in states}
    problem_iterations: list[dict[str, Any]] = []
    latest_state = states[-1] if states else None
    latest_entry: dict[str, Any] | None = None

    for state in states:
        status = classify_state_status(state=state, run_root=run_root)
        parent_iteration = parent_iteration_for_state(state=state)
        parent_state = (
            state_by_iteration.get(parent_iteration)
            if parent_iteration is not None
            else None
        )
        delta_entry = _score_delta_entry(
            state=state,
            parent_state=parent_state,
            best_completed_state=best_completed_state,
        )
        artifact_paths = _artifact_path_map(state=state, run_root=run_root)
        failed_tasks = _problem_task_entries(
            state=state,
            artifact_paths=artifact_paths,
        )
        entry = {
            "iteration": state.iteration,
            "status": status,
            "branchable": state.result is not None,
            "baseline": state.baseline,
            "parent_iteration": parent_iteration,
            "workspace_path": state.refiner_workspace_path,
            "state_path": state.path,
            "planner_selected_iteration": state.planner_selected_iteration,
            "planner_selected_state_index": state.planner_selected_state_index,
            "plan_summary": _plan_summary(plan=state.plan),
            "plan": state.plan,
            "notes_excerpt": _notes_excerpt(notes=state.notes),
            "scores": _score_payload(result=state.result),
            "score_deltas": delta_entry,
            "artifact_paths": artifact_paths,
            "failed_tasks": failed_tasks,
            "created_at_utc": state.created_at_utc,
        }
        iteration_entries.append(entry)
        if latest_state and state.iteration == latest_state.iteration:
            latest_entry = entry
        if failed_tasks:
            failed_iterations.append(
                {
                    "iteration": state.iteration,
                    "status": status,
                    "problem_tasks": failed_tasks,
                    "benchmark_stdout_path": artifact_paths.get(
                        "benchmark_stdout_path"
                    ),
                    "benchmark_stderr_path": artifact_paths.get(
                        "benchmark_stderr_path"
                    ),
                    "aggregate_result_path": artifact_paths.get(
                        "aggregate_result_path"
                    ),
                    "harbor_job_dir": artifact_paths.get("harbor_job_dir"),
                }
            )
            problem_iterations.append(entry)
        if delta_entry:
            delta_entries.append(delta_entry)

    planning_context_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "iteration_count": len(states),
        "completed_iteration_count": len(completed_states),
        "candidate_state_count": len(completed_states),
        "best_completed_iteration": (
            best_completed_state.iteration if best_completed_state else None
        ),
        "latest_iteration": latest_entry,
        "iterations": iteration_entries,
    }
    failed_task_index_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "iterations_with_problem_tasks": failed_iterations,
    }
    return PlannerContextBundle(
        planning_context_payload=planning_context_payload,
        plan_history_text=_render_plan_history(
            iterations=iteration_entries,
        ),
        failed_task_index_payload=failed_task_index_payload,
        failed_task_trajectories_text=_render_failed_task_trajectories(
            run_root=run_root,
            iterations=problem_iterations,
        ),
        score_analysis_text=_render_score_analysis(
            iterations=iteration_entries,
            delta_entries=delta_entries,
            best_completed_state=best_completed_state,
            latest_state=latest_state,
        ),
    )


def classify_state_status(*, state: AgentState, run_root: Path) -> str:
    if state.result is not None and state.official_benchmark is not None:
        return "completed"
    iteration_artifacts = _iteration_artifacts_dir(
        run_root=run_root,
        iteration=state.iteration,
    )
    benchmark_step = _load_step_result(
        path=iteration_artifacts / "benchmark_step.json",
    )
    if benchmark_step and benchmark_step.get("returncode") not in (None, 0):
        return "benchmark_failed"
    validation_step = _load_step_result(
        path=iteration_artifacts / "validation_step.json",
    )
    if validation_step and validation_step.get("returncode") not in (None, 0):
        return "validation_failed"
    implementation_step = _load_step_result(
        path=iteration_artifacts / "implementation_step.json",
    )
    if implementation_step and implementation_step.get("returncode") not in (None, 0):
        return "implementation_failed"
    if benchmark_step:
        return "benchmark_pending_summary"
    if validation_step:
        return "benchmark_pending"
    if implementation_step:
        return "validation_pending"
    if state.plan:
        return "planned"
    return "seeded"


def parent_iteration_for_state(*, state: AgentState) -> int | None:
    if state.prev_path is None:
        return None
    stem = Path(state.prev_path).stem
    if not stem.startswith("iteration-"):
        return None
    suffix = stem.removeprefix("iteration-")
    try:
        return int(suffix)
    except ValueError:
        return None


def summarize_result_line(*, state: AgentState, run_root: Path) -> str:
    status = classify_state_status(state=state, run_root=run_root)
    result = state.result
    if result is None or result.reward_mean is None:
        return f"status={status}"
    return (
        f"status={status}, reward={result.reward_mean:.3f}, "
        f"passed={result.pass_count}, failed={result.failure_count}, "
        f"errors={result.error_count}"
    )


def summarize_problem_trials(*, state: AgentState) -> str:
    result = state.result
    if result is None:
        return "No benchmark result available."
    trial_ids = _problem_trial_ids(result=result)
    if not trial_ids:
        return "No failed or erroring tasks recorded."
    preview = ", ".join(trial_ids[:5])
    if len(trial_ids) > 5:
        preview += ", ..."
    return preview


def planner_notes_template() -> str:
    return "\n".join(
        [
            "# Planner Notes",
            "",
            "Use this file to avoid repeating the same ideas.",
            "- Keep every prior iteration section.",
            "- For each iteration, maintain 1-2 bullets under `Plan`, `Result`, and `Reflection`.",
            "- Replace placeholder bullets with concrete observations once you inspect the evidence.",
            "",
        ]
    )


def latest_iteration_section(*, state: AgentState, run_root: Path) -> str:
    result = state.result
    problem_preview = summarize_problem_trials(state=state)
    reward_line = "N/A"
    if result and result.reward_mean is not None:
        reward_line = (
            f"{result.reward_mean:.3f} / {result.pass_count} / "
            f"{result.failure_count} / {result.error_count}"
        )
    return "\n".join(
        [
            f"## Iteration {state.iteration}",
            "",
            "### Snapshot",
            f"- Current status: {classify_state_status(state=state, run_root=run_root)}",
            f"- Reward / passed / failed / errors: {reward_line}",
            f"- Parent iteration: {parent_iteration_for_state(state=state)}",
            f"- Problem tasks to inspect: {problem_preview}",
            "- Inspect the latest result files before choosing the next parent state.",
            "",
            "### Plan",
            "- Record the main change this iteration tried.",
            "- Record why that change seemed promising at the time.",
            "",
            "### Result",
            "- Record the most important quantitative result after inspecting the latest scores.",
            "- Record the most important qualitative result after inspecting failed-task evidence.",
            "",
            "### Reflection",
            "- Record what should be repeated or built on next.",
            "- Record what should not be retried without a different hypothesis.",
            "",
        ]
    )


def latest_iteration_header(*, state: AgentState) -> str:
    return f"## Iteration {state.iteration}"


def _best_completed_state(*, states: list[AgentState]) -> AgentState | None:
    completed = [
        state
        for state in states
        if state.result is not None and state.result.reward_mean is not None
    ]
    if not completed:
        return None
    return max(
        completed,
        key=lambda state: (
            state.result.reward_mean if state.result else float("-inf"),
            state.result.pass_count if state.result else -1,
            -(state.result.error_count if state.result else 0),
            -(state.result.failure_count if state.result else 0),
            -state.iteration,
        ),
    )


def _render_plan_history(*, iterations: list[dict[str, Any]]) -> str:
    lines = [
        "# Plan History",
        "",
        "| Iter | Status | Parent | Reward | Plan Summary |",
        "|------|--------|--------|--------|--------------|",
    ]
    for entry in iterations:
        lines.append(
            "| {iteration} | {status} | {parent} | {reward} | {summary} |".format(
                iteration=entry["iteration"],
                status=entry["status"],
                parent=entry["parent_iteration"]
                if entry["parent_iteration"] is not None
                else "root",
                reward=_reward_text(entry=entry),
                summary=_table_safe_text(
                    text=entry["plan_summary"] or "No plan recorded"
                ),
            )
        )
    lines.extend(["", "## Full Plans", ""])
    for entry in iterations:
        lines.extend(
            [
                f"### Iteration {entry['iteration']}",
                f"- Status: {entry['status']}",
                f"- Parent iteration: {entry['parent_iteration'] if entry['parent_iteration'] is not None else 'root'}",
                f"- Reward: {_reward_text(entry=entry)}",
                f"- Plan summary: {entry['plan_summary'] or 'No plan recorded'}",
                "",
                "Plan:",
                "",
            ]
        )
        plan = entry["plan"] or "No plan recorded."
        for raw_line in str(plan).splitlines():
            lines.append(f"> {raw_line}" if raw_line else ">")
        lines.extend([""])
    return "\n".join(lines).rstrip() + "\n"


def _render_failed_task_trajectories(
    *, run_root: Path, iterations: list[dict[str, Any]]
) -> str:
    lines = [
        "# Failed Task Trajectories",
        "",
        "Use this file for qualitative analysis before choosing the next parent state.",
        "- For the parent state you are considering, inspect its failed-task artifacts first.",
        "- Prefer the listed trajectory, verifier, and result files over guessing from aggregate metrics alone.",
        "",
    ]
    if not iterations:
        lines.append(
            "No failed or erroring tasks were recorded in the persisted iterations."
        )
        return "\n".join(lines) + "\n"
    for entry in iterations:
        lines.extend(
            [
                f"## Iteration {entry['iteration']}",
                f"- Status: {entry['status']}",
                f"- Reward / passed / failed / errors: {_reward_text(entry=entry)} / "
                f"{_score_count(entry=entry, key='pass_count')} / "
                f"{_score_count(entry=entry, key='failure_count')} / "
                f"{_score_count(entry=entry, key='error_count')}",
                f"- Harbor job dir: {entry['artifact_paths'].get('harbor_job_dir') or 'N/A'}",
                "",
            ]
        )
        for task in entry["failed_tasks"]:
            lines.extend(
                [
                    f"### {task['trial_id']}",
                    f"- Task name: {task['task_name']}",
                    f"- Reward failed: {task['reward_failed']}",
                    f"- Exception types: {', '.join(task['exception_types']) if task['exception_types'] else 'none recorded'}",
                    f"- Trial directory: {task['trial_dir'] or 'not found'}",
                ]
            )
            if task["artifact_paths"]:
                lines.append("- Suggested artifact paths:")
                for path in task["artifact_paths"]:
                    lines.append(f"  - {path}")
            else:
                lines.append(
                    "- Suggested artifact paths: none discovered; fall back to aggregate result and benchmark logs."
                )
            if task["preview"]:
                lines.extend(["", "Preview:", "```text", task["preview"], "```"])
            lines.append("")
        lines.extend(
            [
                f"- Benchmark stdout: {entry['artifact_paths'].get('benchmark_stdout_path') or 'N/A'}",
                f"- Benchmark stderr: {entry['artifact_paths'].get('benchmark_stderr_path') or 'N/A'}",
                f"- Aggregate result: {entry['artifact_paths'].get('aggregate_result_path') or 'N/A'}",
                "",
            ]
        )
    lines.append(f"_Generated from run root: {run_root}_")
    return "\n".join(lines).rstrip() + "\n"


def _render_score_analysis(
    *,
    iterations: list[dict[str, Any]],
    delta_entries: list[dict[str, Any]],
    best_completed_state: AgentState | None,
    latest_state: AgentState | None,
) -> str:
    lines = [
        "# Quantitative Score Analysis",
        "",
        f"- Persisted iterations: {len(iterations)}",
        f"- Completed iterations: {sum(1 for entry in iterations if entry['branchable'])}",
        (
            f"- Best completed iteration: {best_completed_state.iteration}"
            if best_completed_state
            else "- Best completed iteration: N/A"
        ),
        (
            f"- Latest iteration: {latest_state.iteration}"
            if latest_state
            else "- Latest iteration: N/A"
        ),
        "",
        "| Iter | Status | Parent | Reward | dReward(parent) | dPass | dFail | dErr |",
        "|------|--------|--------|--------|-----------------|-------|-------|------|",
    ]
    for entry in iterations:
        deltas = entry["score_deltas"] or {}
        lines.append(
            "| {iteration} | {status} | {parent} | {reward} | {reward_delta} | {pass_delta} | {failure_delta} | {error_delta} |".format(
                iteration=entry["iteration"],
                status=entry["status"],
                parent=entry["parent_iteration"]
                if entry["parent_iteration"] is not None
                else "root",
                reward=_reward_text(entry=entry),
                reward_delta=_signed_float_text(
                    value=deltas.get("reward_vs_parent"),
                ),
                pass_delta=_signed_int_text(value=deltas.get("pass_count_vs_parent")),
                failure_delta=_signed_int_text(
                    value=deltas.get("failure_count_vs_parent"),
                ),
                error_delta=_signed_int_text(value=deltas.get("error_count_vs_parent")),
            )
        )
    ranked_deltas = [
        entry for entry in delta_entries if entry.get("reward_vs_parent") is not None
    ]
    largest_improvements = sorted(
        ranked_deltas,
        key=lambda entry: float(entry["reward_vs_parent"]),
        reverse=True,
    )[:3]
    largest_regressions = sorted(
        ranked_deltas,
        key=lambda entry: float(entry["reward_vs_parent"]),
    )[:3]
    lines.extend(["", "## Largest Improvements", ""])
    if largest_improvements:
        for entry in largest_improvements:
            lines.append(_delta_bullet(entry=entry))
    else:
        lines.append("- No parent-vs-child score deltas are available yet.")
    lines.extend(["", "## Largest Regressions", ""])
    if largest_regressions:
        for entry in largest_regressions:
            lines.append(_delta_bullet(entry=entry))
    else:
        lines.append("- No parent-vs-child score deltas are available yet.")
    return "\n".join(lines).rstrip() + "\n"


def _delta_bullet(*, entry: dict[str, Any]) -> str:
    return (
        f"- Iteration {entry['iteration']} vs parent {entry['parent_iteration']}: "
        f"reward {_signed_float_text(value=entry.get('reward_vs_parent'))}, "
        f"passed {_signed_int_text(value=entry.get('pass_count_vs_parent'))}, "
        f"failed {_signed_int_text(value=entry.get('failure_count_vs_parent'))}, "
        f"errors {_signed_int_text(value=entry.get('error_count_vs_parent'))}."
    )


def _score_delta_entry(
    *,
    state: AgentState,
    parent_state: AgentState | None,
    best_completed_state: AgentState | None,
) -> dict[str, Any] | None:
    result = state.result
    if result is None:
        return None
    parent_result = parent_state.result if parent_state else None
    best_result = best_completed_state.result if best_completed_state else None
    return {
        "iteration": state.iteration,
        "parent_iteration": parent_state.iteration if parent_state else None,
        "reward_vs_parent": _float_delta(
            current=result.reward_mean,
            previous=parent_result.reward_mean if parent_result else None,
        ),
        "pass_count_vs_parent": _int_delta(
            current=result.pass_count,
            previous=parent_result.pass_count if parent_result else None,
        ),
        "failure_count_vs_parent": _int_delta(
            current=result.failure_count,
            previous=parent_result.failure_count if parent_result else None,
        ),
        "error_count_vs_parent": _int_delta(
            current=result.error_count,
            previous=parent_result.error_count if parent_result else None,
        ),
        "reward_vs_best": _float_delta(
            current=result.reward_mean,
            previous=best_result.reward_mean if best_result else None,
        ),
    }


def _problem_task_entries(
    *, state: AgentState, artifact_paths: dict[str, str | None]
) -> list[dict[str, Any]]:
    result = state.result
    benchmark = state.official_benchmark
    if result is None:
        return []
    exception_lookup = _exception_lookup(result=result)
    task_entries: list[dict[str, Any]] = []
    for trial_id in _problem_trial_ids(result=result):
        trial_dir = _resolve_trial_dir(
            benchmark=benchmark,
            trial_id=trial_id,
        )
        artifact_paths_for_trial = _trial_artifact_paths(trial_dir=trial_dir)
        task_entries.append(
            {
                "trial_id": trial_id,
                "task_name": _task_name_from_trial_id(trial_id=trial_id),
                "reward_failed": trial_id in result.failed_trials,
                "exception_types": exception_lookup.get(trial_id, []),
                "trial_dir": str(trial_dir) if trial_dir else None,
                "artifact_paths": artifact_paths_for_trial,
                "preview": _trial_preview(trial_dir=trial_dir),
                "aggregate_result_path": artifact_paths.get("aggregate_result_path"),
            }
        )
    return task_entries


def _iteration_artifacts_dir(*, run_root: Path, iteration: int) -> Path:
    return run_root / "artifacts" / f"iteration-{iteration:04d}"


def _artifact_path_map(*, state: AgentState, run_root: Path) -> dict[str, str | None]:
    iteration_artifacts = _iteration_artifacts_dir(
        run_root=run_root,
        iteration=state.iteration,
    )
    benchmark = state.official_benchmark
    return {
        "planner_prompt_artifact_path": _existing_path(
            path=Path(state.planner_prompt_artifact_path)
            if state.planner_prompt_artifact_path
            else None
        ),
        "planner_output_artifact_path": _existing_path(
            path=Path(state.planner_output_artifact_path)
            if state.planner_output_artifact_path
            else None
        ),
        "implementation_step_path": _existing_path(
            path=iteration_artifacts / "implementation_step.json",
        ),
        "validation_step_path": _existing_path(
            path=iteration_artifacts / "validation_step.json",
        ),
        "benchmark_step_path": _existing_path(
            path=iteration_artifacts / "benchmark_step.json",
        ),
        "benchmark_result_path": _existing_path(
            path=iteration_artifacts / "benchmark_result.json",
        ),
        "benchmark_summary_path": _existing_path(
            path=Path(benchmark.benchmark_summary_path) if benchmark else None,
        ),
        "benchmark_stdout_path": _existing_path(
            path=Path(benchmark.benchmark_stdout_path) if benchmark else None,
        ),
        "benchmark_stderr_path": _existing_path(
            path=Path(benchmark.benchmark_stderr_path) if benchmark else None,
        ),
        "aggregate_result_path": _existing_path(
            path=Path(benchmark.aggregate_result_path) if benchmark else None,
        ),
        "harbor_job_dir": _existing_path(
            path=Path(benchmark.harbor_job_dir) if benchmark else None,
        ),
    }


def _existing_path(*, path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return str(path.resolve())


def _load_step_result(*, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _score_payload(result: BenchmarkSummary | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "reward_mean": result.reward_mean,
        "n_trials": result.n_trials,
        "pass_count": result.pass_count,
        "failure_count": result.failure_count,
        "error_count": result.error_count,
        "passed_trials": list(result.passed_trials),
        "failed_trials": list(result.failed_trials),
        "exception_types": dict(result.exception_types),
    }


def _notes_excerpt(*, notes: str | None, max_lines: int = 14) -> str | None:
    if not notes:
        return None
    lines = notes.splitlines()
    excerpt = "\n".join(lines[:max_lines]).strip()
    if len(lines) > max_lines:
        excerpt += "\n..."
    return excerpt or None


def _reward_text(*, entry: dict[str, Any]) -> str:
    scores = entry.get("scores") or {}
    reward = scores.get("reward_mean")
    if reward is None:
        return "N/A"
    return f"{float(reward):.3f}"


def _score_count(*, entry: dict[str, Any], key: str) -> str:
    scores = entry.get("scores") or {}
    value = scores.get(key)
    return str(value) if value is not None else "N/A"


def _signed_float_text(*, value: object) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value):+.3f}"


def _signed_int_text(*, value: object) -> str:
    if not isinstance(value, int):
        return "N/A"
    return f"{value:+d}"


def _float_delta(*, current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _int_delta(*, current: int | None, previous: int | None) -> int | None:
    if current is None or previous is None:
        return None
    return current - previous


def _plan_summary(*, plan: str | None) -> str | None:
    if not plan:
        return None
    first_line = next((line.strip() for line in plan.splitlines() if line.strip()), "")
    return first_line or None


def _table_safe_text(*, text: str) -> str:
    return text.replace("|", "/")


def _problem_trial_ids(*, result: BenchmarkSummary) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for trial_id in list(result.failed_trials) + [
        trial_id
        for trial_ids in result.exception_types.values()
        for trial_id in trial_ids
    ]:
        if trial_id in seen:
            continue
        seen.add(trial_id)
        ordered.append(trial_id)
    return ordered


def _exception_lookup(*, result: BenchmarkSummary) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for exception_type, trial_ids in result.exception_types.items():
        for trial_id in trial_ids:
            mapping.setdefault(trial_id, []).append(exception_type)
    return mapping


def _resolve_trial_dir(
    *, benchmark: OfficialBenchmarkRun | None, trial_id: str
) -> Path | None:
    if benchmark is None:
        return None
    harbor_job_dir = Path(benchmark.harbor_job_dir)
    direct = harbor_job_dir / trial_id
    if direct.exists():
        return direct
    for candidate in harbor_job_dir.rglob(trial_id):
        if candidate.is_dir():
            return candidate
    return None


def _trial_artifact_paths(*, trial_dir: Path | None) -> list[str]:
    if trial_dir is None:
        return []
    collected: list[str] = []
    for relative_path in _TRIAL_ARTIFACT_RELATIVE_PATHS:
        candidate = trial_dir / relative_path
        if candidate.exists():
            collected.append(str(candidate.resolve()))
    return collected


def _trial_preview(*, trial_dir: Path | None) -> str | None:
    if trial_dir is None:
        return None
    previews: list[str] = []
    result_path = trial_dir / "result.json"
    result_preview = _trial_result_preview(path=result_path)
    if result_preview:
        previews.append(result_preview)
    for relative_path in (
        Path("agent") / "trajectory.json",
        Path("verifier") / "test-stdout.txt",
        Path("verifier") / "test-stderr.txt",
    ):
        candidate = trial_dir / relative_path
        preview = _file_preview(path=candidate)
        if preview:
            previews.append(f"{relative_path}:\n{preview}")
    if not previews:
        return None
    return "\n\n".join(previews)


def _trial_result_preview(*, path: Path) -> str | None:
    data = _load_json_dict(path=path)
    if not data:
        return None
    lines: list[str] = []
    task_name = _string_value(value=data.get("task_name"))
    if task_name:
        lines.append(f"task_name: {task_name}")
    agent_stdout = _nested_string(
        data=data,
        path=("agent_result", "metadata", "small_agent_result", "stdout"),
    )
    if agent_stdout:
        lines.append(f"agent stdout: {_compact_text(text=agent_stdout)}")
    agent_stderr = _nested_string(
        data=data,
        path=("agent_result", "metadata", "small_agent_result", "stderr"),
    )
    if agent_stderr:
        lines.append(f"agent stderr: {_compact_text(text=agent_stderr)}")
    return "\n".join(lines) if lines else None


def _file_preview(*, path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    return _compact_text(text=content)


def _compact_text(*, text: str, max_chars: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _load_json_dict(*, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested_string(*, data: dict[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _string_value(value=current)


def _string_value(*, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _task_name_from_trial_id(*, trial_id: str) -> str:
    if "__" not in trial_id:
        return trial_id
    return trial_id.split("__", maxsplit=1)[0]
