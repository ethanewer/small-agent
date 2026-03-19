# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from pathlib import Path
import re

from agent_evolve_v3.services.benchmark import TrialLogSummary, load_trial_summaries
from agent_evolve_v3.state.types import AgentState, BenchmarkSummary

PLANNER_NOTES_FILE_NAME = "PLANNER_NOTES.md"


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


def plan_summary(*, plan: str | None) -> str:
    if not plan:
        return "No plan recorded."
    first_line = next((line.strip() for line in plan.splitlines() if line.strip()), "")
    return first_line or "No plan recorded."


def planner_notes_template() -> str:
    return "\n".join(
        [
            "# Planner Notes",
            "",
            "Use this file to avoid repeating the same ideas.",
            "- Keep every prior iteration section.",
            "- For each iteration, maintain a couple bullets under `Plan`, `Result`, and `Reflection`.",
            "- When you choose a new plan, update the upcoming iteration section's `Plan` bullets immediately.",
            "- On the next planning pass, update that same iteration section's `Result` and `Reflection` after the plan has been implemented and evaluated.",
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
            f"- Plan summary: {plan_summary(plan=state.plan)}",
            f"- Problem tasks to inspect: {problem_preview}",
            "- Inspect the latest result files before choosing the next parent state.",
            "",
            "### Plan",
            "- Record the main change planned for this iteration as soon as you choose it.",
            "- Record why that change seemed promising at planning time.",
            "",
            "### Result",
            "- On the next planning pass, record the most important quantitative result after this plan was implemented and evaluated.",
            "- On the next planning pass, record the most important qualitative result after inspecting failed-task evidence.",
            "",
            "### Reflection",
            "- On the next planning pass, record what should be repeated or built on next.",
            "- On the next planning pass, record what should not be retried without a different hypothesis.",
            "",
        ]
    )


def latest_iteration_header(*, state: AgentState) -> str:
    return f"## Iteration {state.iteration}"


def latest_run_selectable_text(*, state: AgentState | None) -> str:
    if state is None:
        return "N/A"
    return "yes" if state.result is not None else "no"


def latest_run_parent_iteration_text(*, state: AgentState | None) -> str:
    if state is None:
        return "N/A"
    parent_iteration = parent_iteration_for_state(state=state)
    return str(parent_iteration) if parent_iteration is not None else "root"


def latest_run_artifact_map(
    *, state: AgentState | None, run_root: Path
) -> dict[str, str]:
    if state is None:
        return {
            "benchmark_summary_path": "N/A",
            "benchmark_stdout_path": "N/A",
            "benchmark_stderr_path": "N/A",
            "harbor_job_dir": "N/A",
            "implementation_step_path": "N/A",
            "validation_step_path": "N/A",
            "benchmark_step_path": "N/A",
            "benchmark_result_path": "N/A",
            "trial_summaries_path": "N/A",
            "trial_logs_dir": "N/A",
        }
    benchmark = state.official_benchmark
    iteration_artifacts = _iteration_artifacts_dir(
        run_root=run_root,
        iteration=state.iteration,
    )
    return {
        "benchmark_summary_path": _existing_path(
            path=Path(benchmark.benchmark_summary_path) if benchmark else None,
        )
        or "N/A",
        "benchmark_stdout_path": _existing_path(
            path=Path(benchmark.benchmark_stdout_path) if benchmark else None,
        )
        or "N/A",
        "benchmark_stderr_path": _existing_path(
            path=Path(benchmark.benchmark_stderr_path) if benchmark else None,
        )
        or "N/A",
        "harbor_job_dir": _existing_path(
            path=Path(benchmark.harbor_job_dir) if benchmark else None,
        )
        or "N/A",
        "implementation_step_path": _existing_path(
            path=iteration_artifacts / "implementation_step.json",
        )
        or "N/A",
        "validation_step_path": _existing_path(
            path=iteration_artifacts / "validation_step.json",
        )
        or "N/A",
        "benchmark_step_path": _existing_path(
            path=iteration_artifacts / "benchmark_step.json",
        )
        or "N/A",
        "benchmark_result_path": _existing_path(
            path=iteration_artifacts / "benchmark_result.json",
        )
        or "N/A",
        "trial_summaries_path": _existing_path(
            path=(
                Path(benchmark.trial_summaries_path)
                if benchmark and benchmark.trial_summaries_path
                else None
            ),
        )
        or "N/A",
        "trial_logs_dir": _existing_path(
            path=(
                Path(benchmark.trial_logs_dir)
                if benchmark and benchmark.trial_logs_dir
                else None
            ),
        )
        or "N/A",
    }


_MAX_INLINE_TRIALS = 10
_INLINE_STDOUT_LIMIT = 500
_INLINE_VERIFIER_LIMIT = 300


def summarize_problem_trials_detail(*, state: AgentState | None) -> str:
    if state is None:
        return "No benchmark data available."
    benchmark = state.official_benchmark
    if benchmark is None:
        return "No benchmark data available."
    result = state.result
    if result is None:
        return "No benchmark result available."

    trial_summaries_path = benchmark.trial_summaries_path
    if not trial_summaries_path:
        return _fallback_problem_summary(result=result)

    summaries = load_trial_summaries(path=Path(trial_summaries_path))
    if not summaries:
        return _fallback_problem_summary(result=result)

    problem_ids = set(_problem_trial_ids(result=result))
    if not problem_ids:
        return "All trials passed."

    problem_summaries = [s for s in summaries if s.trial_name in problem_ids]
    if not problem_summaries:
        return _fallback_problem_summary(result=result)

    trial_logs_dir = benchmark.trial_logs_dir
    lines: list[str] = []
    shown = problem_summaries[:_MAX_INLINE_TRIALS]
    for summary in shown:
        lines.append(
            _format_trial_detail(
                summary=summary,
                trial_logs_dir=trial_logs_dir,
            )
        )

    remaining = len(problem_summaries) - len(shown)
    if remaining > 0:
        lines.append(
            f"... and {remaining} more problem trial(s)."
            + f" See `{trial_summaries_path}` for full details."
        )

    return "\n\n".join(lines)


def _format_trial_detail(
    *,
    summary: TrialLogSummary,
    trial_logs_dir: str | None = None,
) -> str:
    reward_str = f"{summary.reward:.1f}" if summary.reward is not None else "N/A"
    exit_str = (
        str(summary.agent_exit_code) if summary.agent_exit_code is not None else "N/A"
    )

    header = f"### {summary.task_name} (reward={reward_str})"
    status_parts = [f"Agent exit: {exit_str}"]
    if summary.exception_type:
        status_parts.append(f"Exception: {summary.exception_type}")
    status_line = " | ".join(status_parts)

    sections = [header, status_line]

    if trial_logs_dir:
        safe_name = re.sub(r"[^\w\-.]", "_", summary.task_name)
        sections.append(f"Full agent log: `{trial_logs_dir}/{safe_name}.txt`")
    elif summary.agent_stdout_tail:
        tail = _truncate_text(
            text=summary.agent_stdout_tail,
            limit=_INLINE_STDOUT_LIMIT,
        )
        sections.append(f"Agent log tail:\n```\n{tail}\n```")

    if summary.verifier_summary:
        verifier = _truncate_text(
            text=summary.verifier_summary,
            limit=_INLINE_VERIFIER_LIMIT,
        )
        sections.append(f"Verifier output:\n```\n{verifier}\n```")

    return "\n".join(sections)


def _truncate_text(*, text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text.strip()
    return "..." + encoded[-limit:].decode("utf-8", errors="ignore").strip()


def _fallback_problem_summary(*, result: BenchmarkSummary) -> str:
    trial_ids = _problem_trial_ids(result=result)
    if not trial_ids:
        return "No failed or erroring tasks recorded."
    preview = ", ".join(trial_ids[:5])
    if len(trial_ids) > 5:
        preview += ", ..."
    return f"Problem trials (IDs only, no detailed logs available): {preview}"


def _iteration_artifacts_dir(*, run_root: Path, iteration: int) -> Path:
    return run_root / "artifacts" / f"iteration-{iteration:04d}"


def _existing_path(*, path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return str(path.resolve())


def _load_step_result(*, path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload_obj, dict):
        return None
    return {str(key): value for key, value in payload_obj.items()}


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
