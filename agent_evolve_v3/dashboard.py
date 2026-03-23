# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportCallIssue=false, reportUnusedFunction=false

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from agent_evolve_v3.state.planner_context import (
    classify_state_status,
    parent_iteration_for_state,
    plan_summary,
)
from agent_evolve_v3.state.types import AgentState

app = FastAPI()

OUTPUTS_ROOT = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Trial outcome classification
# ---------------------------------------------------------------------------

OUTCOME_PASS = "pass"
OUTCOME_FAILURE = "failure"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_MAX_ITERS = "max_iters"
OUTCOME_OTHER_ERROR = "other_error"
OUTCOME_PENDING = "pending"


def _classify_trial(trial: dict[str, Any]) -> str:
    verifier = trial.get("verifier_result")
    if verifier and isinstance(verifier, dict):
        rewards = verifier.get("rewards", {})
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
            if reward == 1.0:
                return OUTCOME_PASS

    exc = trial.get("exception_info")
    if exc and isinstance(exc, dict):
        exc_type = exc.get("exception_type", "")
        if exc_type == "AgentTimeoutError":
            return OUTCOME_TIMEOUT
        return OUTCOME_OTHER_ERROR

    if verifier and isinstance(verifier, dict):
        rewards = verifier.get("rewards", {})
        if isinstance(rewards, dict) and rewards.get("reward") is not None:
            return OUTCOME_FAILURE

    return OUTCOME_PENDING


def _task_name_from_trial_name(trial_name: str) -> str:
    parts = trial_name.rsplit("__", 1)
    return parts[0] if len(parts) == 2 else trial_name


# ---------------------------------------------------------------------------
# Harbor job dir scanning (two-tier aware)
# ---------------------------------------------------------------------------


def _find_all_harbor_job_dirs(*, artifacts_dir: Path) -> list[Path]:
    """Find all harbor job dirs across train_small, train_remaining, train_full,
    and the legacy official_benchmark layout."""
    dirs: list[Path] = []

    for phase in ("train_small", "train_remaining", "train_full"):
        phase_dir = artifacts_dir / phase
        if not phase_dir.is_dir():
            continue
        jobs_dir = phase_dir / "harbor_jobs"
        if not jobs_dir.is_dir():
            continue
        for job_dir in sorted(jobs_dir.iterdir()):
            if job_dir.is_dir() and (job_dir / "result.json").exists():
                dirs.append(job_dir)

    legacy = artifacts_dir / "official_benchmark" / "harbor_jobs"
    if legacy.is_dir():
        for job_dir in sorted(legacy.iterdir()):
            if job_dir.is_dir() and (job_dir / "result.json").exists():
                dirs.append(job_dir)

    return dirs


def _find_harbor_job_dir(*, artifacts_dir: Path) -> Path | None:
    """Return the latest harbor job dir (for backward compat)."""
    dirs = _find_all_harbor_job_dirs(artifacts_dir=artifacts_dir)
    return dirs[-1] if dirs else None


def _load_aggregate_result(*, harbor_job_dir: Path) -> dict[str, Any] | None:
    result_path = harbor_job_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        data: object = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _load_trial_results(*, harbor_job_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not harbor_job_dir.is_dir():
        return results
    for child in harbor_job_dir.iterdir():
        if not child.is_dir() or "__" not in child.name:
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            data: object = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            results[child.name] = data

    return results


def _load_all_trial_results(*, artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Load trial results from all harbor job dirs in the artifacts."""
    combined: dict[str, dict[str, Any]] = {}
    for job_dir in _find_all_harbor_job_dirs(artifacts_dir=artifacts_dir):
        combined.update(_load_trial_results(harbor_job_dir=job_dir))
    return combined


# ---------------------------------------------------------------------------
# Benchmark progress for two-tier flow
# ---------------------------------------------------------------------------


def _get_two_tier_benchmark_progress(*, artifacts_dir: Path) -> dict[str, Any] | None:
    """Return progress info for the current benchmark phase."""
    progress: dict[str, Any] = {"phases": []}

    for phase in ("train_small", "train_remaining", "train_full"):
        phase_dir = artifacts_dir / phase
        if not phase_dir.is_dir():
            continue

        summary_path = phase_dir / "benchmark_summary.json"
        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                progress["phases"].append(
                    {
                        "phase": phase,
                        "status": "done",
                        "completed": s.get("n_trials", 0),
                        "total": s.get("n_trials", 0),
                    }
                )
            except (json.JSONDecodeError, OSError):
                pass
            continue

        jobs_dir = phase_dir / "harbor_jobs"
        if not jobs_dir.is_dir():
            continue
        for job_dir in sorted(jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            agg = _load_aggregate_result(harbor_job_dir=job_dir)
            if agg is None:
                continue
            n_total = agg.get("n_total_trials", 0)
            n_done = agg.get("stats", {}).get("n_trials", 0)
            progress["phases"].append(
                {
                    "phase": phase,
                    "status": "running" if n_done < n_total else "done",
                    "completed": n_done,
                    "total": n_total,
                }
            )

    if not progress["phases"]:
        return None
    return progress


# ---------------------------------------------------------------------------
# Iteration-level metrics
# ---------------------------------------------------------------------------


def _compute_iteration_metrics(
    *, trial_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    task_outcomes: dict[str, str] = {}
    for trial_name, trial_data in trial_results.items():
        task = _task_name_from_trial_name(trial_name)
        task_outcomes[task] = _classify_trial(trial_data)

    counts = {
        OUTCOME_PASS: 0,
        OUTCOME_FAILURE: 0,
        OUTCOME_TIMEOUT: 0,
        OUTCOME_MAX_ITERS: 0,
        OUTCOME_OTHER_ERROR: 0,
    }
    for outcome in task_outcomes.values():
        if outcome in counts:
            counts[outcome] += 1

    n_classified = sum(counts.values())
    total = n_classified if n_classified > 0 else 1

    return {
        "completion_rate": counts[OUTCOME_PASS] / total,
        "failure_rate": counts[OUTCOME_FAILURE] / total,
        "timeout_rate": counts[OUTCOME_TIMEOUT] / total,
        "max_iters_rate": counts[OUTCOME_MAX_ITERS] / total,
        "other_error_rate": counts[OUTCOME_OTHER_ERROR] / total,
        "timeout_or_turn_limit_rate": (
            counts[OUTCOME_TIMEOUT] + counts[OUTCOME_MAX_ITERS]
        )
        / total,
        "tasks": task_outcomes,
    }


def _compute_metrics_from_state(*, state: AgentState) -> dict[str, Any] | None:
    """Classify each trial and compute per-trial outcome rates."""
    result = state.result
    if result is None or result.reward_mean is None:
        return None

    passed_set = set(result.passed_trials)
    exc_types = result.exception_types
    if not exc_types and result.sample_results:
        exc_types = {}
        for s in result.sample_results:
            for exc_type, tids in s.exception_types.items():
                exc_types.setdefault(exc_type, []).extend(tids)

    error_map: dict[str, str] = {}
    for exc_type, trial_ids in exc_types.items():
        for tid in trial_ids:
            error_map[tid] = exc_type

    trial_outcomes: dict[str, str] = {}
    all_trial_ids = (
        set(result.passed_trials) | set(result.failed_trials) | set(error_map)
    )
    for tid in all_trial_ids:
        if tid in passed_set:
            trial_outcomes[tid] = OUTCOME_PASS
        elif tid in error_map:
            exc = error_map[tid]
            if exc == "AgentTimeoutError":
                trial_outcomes[tid] = OUTCOME_TIMEOUT
            else:
                trial_outcomes[tid] = OUTCOME_OTHER_ERROR
        else:
            trial_outcomes[tid] = OUTCOME_FAILURE

    counts = {
        OUTCOME_PASS: 0,
        OUTCOME_FAILURE: 0,
        OUTCOME_TIMEOUT: 0,
        OUTCOME_MAX_ITERS: 0,
        OUTCOME_OTHER_ERROR: 0,
    }
    for outcome in trial_outcomes.values():
        if outcome in counts:
            counts[outcome] += 1

    total = sum(counts.values()) or 1

    task_outcomes: dict[str, str] = {}
    for tid, outcome in trial_outcomes.items():
        task_outcomes[_task_name_from_trial_name(tid)] = outcome

    return {
        "completion_rate": counts[OUTCOME_PASS] / total,
        "failure_rate": counts[OUTCOME_FAILURE] / total,
        "timeout_rate": counts[OUTCOME_TIMEOUT] / total,
        "max_iters_rate": counts[OUTCOME_MAX_ITERS] / total,
        "other_error_rate": counts[OUTCOME_OTHER_ERROR] / total,
        "timeout_or_turn_limit_rate": (
            counts[OUTCOME_TIMEOUT] + counts[OUTCOME_MAX_ITERS]
        )
        / total,
        "tasks": task_outcomes,
    }


# ---------------------------------------------------------------------------
# Stage detection and timing (two-tier aware)
# ---------------------------------------------------------------------------

_STAGE_FILES = [
    ("planner_step.json", "planning"),
    ("implementation_step.json", "implementing"),
    ("validation_step.json", "validating"),
]


def _detect_current_stage(*, artifacts_dir: Path, is_bootstrap: bool = False) -> str:
    if not artifacts_dir.exists():
        return "waiting"

    if is_bootstrap:
        has_result = _has_benchmark_results(artifacts_dir=artifacts_dir)
        if has_result:
            return "completed"
        has_harbor = bool(_find_all_harbor_job_dirs(artifacts_dir=artifacts_dir))
        if has_harbor or any(
            (artifacts_dir / p).is_dir()
            for p in ("train_small", "train_remaining", "train_full")
        ):
            return "benchmarking"
        return "waiting"

    for filename, stage_name in _STAGE_FILES:
        if not (artifacts_dir / filename).exists():
            return stage_name

    has_result = _has_benchmark_results(artifacts_dir=artifacts_dir)
    if has_result:
        return "completed"

    if any(
        (artifacts_dir / p).is_dir()
        for p in ("train_small", "train_remaining", "train_full")
    ):
        return "benchmarking"

    return "benchmarking"


def _has_benchmark_results(*, artifacts_dir: Path) -> bool:
    """Check if any benchmark phase has completed summary files."""
    for phase in ("train_small", "train_remaining", "train_full"):
        phase_dir = artifacts_dir / phase
        if not phase_dir.is_dir():
            continue
        if (phase_dir / "benchmark_summary.json").exists():
            return True

    legacy = artifacts_dir / "official_benchmark" / "harbor_jobs"
    if legacy.is_dir():
        for job_dir in legacy.iterdir():
            if job_dir.is_dir() and (job_dir / "result.json").exists():
                return True
    return False


def _get_benchmark_progress(
    *,
    artifacts_dir: Path,
    n_benchmark_tasks: int = 0,
    n_train_small_tasks: int = 0,
    n_samples: int = 1,
) -> dict[str, int] | None:
    """Compute overall benchmark progress with expected totals."""
    progress = _get_two_tier_benchmark_progress(artifacts_dir=artifacts_dir)
    if progress is None:
        return None
    phases = progress.get("phases", [])
    total_done = sum(p.get("completed", 0) for p in phases)

    has_remaining = any(p.get("phase") == "train_remaining" for p in phases)
    has_full = any(p.get("phase") == "train_full" for p in phases)
    if has_remaining or has_full:
        expected_total = n_benchmark_tasks * n_samples
    else:
        expected_total = n_train_small_tasks * n_samples

    observed_total = sum(p.get("total", 0) for p in phases)
    total_all = max(expected_total, observed_total)
    return {"completed": total_done, "total": total_all}


def _read_live_benchmark_scores(
    *,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Read in-progress benchmark scores from artifact summaries."""
    out: dict[str, Any] = {
        "train_small_reward": None,
        "train_full_reward": None,
        "promoted": False,
    }

    small_summary = artifacts_dir / "train_small" / "benchmark_summary.json"
    if small_summary.exists():
        try:
            d = json.loads(small_summary.read_text(encoding="utf-8"))
            r = d.get("reward_mean")
            if isinstance(r, (int, float)):
                out["train_small_reward"] = float(r)
        except (json.JSONDecodeError, OSError):
            pass

    has_remaining = (artifacts_dir / "train_remaining").is_dir()
    has_full = (artifacts_dir / "train_full").is_dir()
    out["promoted"] = has_remaining or has_full

    return out


def _parse_iso_timestamp(ts: str | None) -> float | None:
    if ts is None:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            return dt.timestamp()
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime if path.exists() else None
    except OSError:
        return None


def _compute_stage_times(
    *,
    artifacts_dir: Path,
    is_bootstrap: bool,
    current_stage: str,
) -> dict[str, float | None]:
    now = time.time()

    if is_bootstrap:
        bench_sec: float | None = None
        bench_live = False
        first_harbor = _find_first_harbor_start(artifacts_dir=artifacts_dir)
        if first_harbor is not None:
            last_summary = _find_latest_summary_mtime(artifacts_dir=artifacts_dir)
            if last_summary is not None and current_stage == "completed":
                bench_sec = last_summary - first_harbor
            else:
                bench_sec = now - first_harbor
                bench_live = current_stage == "benchmarking"

        return {
            "plan_time_sec": None,
            "impl_time_sec": None,
            "benchmark_time_sec": bench_sec,
            "plan_start_epoch": None,
            "impl_start_epoch": None,
            "benchmark_start_epoch": (now - bench_sec)
            if bench_live and bench_sec is not None
            else None,
        }

    prompt_mtime = _file_mtime(artifacts_dir / "planner_prompt.txt")
    planner_mtime = _file_mtime(artifacts_dir / "planner_step.json")
    impl_mtime = _file_mtime(artifacts_dir / "implementation_step.json")
    validation_mtime = _file_mtime(artifacts_dir / "validation_step.json")

    plan_sec: float | None = None
    plan_live = False
    impl_sec: float | None = None
    impl_live = False
    bench_sec = None
    bench_live = False

    if prompt_mtime is not None:
        if planner_mtime is not None:
            plan_sec = planner_mtime - prompt_mtime
        elif current_stage == "planning":
            plan_sec = now - prompt_mtime
            plan_live = True

    if planner_mtime is not None:
        if impl_mtime is not None:
            impl_sec = impl_mtime - planner_mtime
        elif current_stage == "implementing":
            impl_sec = now - planner_mtime
            impl_live = True

    if validation_mtime is not None:
        last_summary = _find_latest_summary_mtime(artifacts_dir=artifacts_dir)
        if last_summary is not None and current_stage == "completed":
            bench_sec = last_summary - validation_mtime
        else:
            bench_sec = now - validation_mtime
            bench_live = current_stage == "benchmarking"

    if plan_sec is not None and plan_sec < 0:
        plan_sec = None
        plan_live = False

    if impl_sec is not None and impl_sec < 0:
        impl_sec = None
        impl_live = False

    if bench_sec is not None and bench_sec < 0:
        bench_sec = None
        bench_live = False

    return {
        "plan_time_sec": plan_sec,
        "impl_time_sec": impl_sec,
        "benchmark_time_sec": bench_sec,
        "plan_start_epoch": (now - plan_sec)
        if plan_live and plan_sec is not None
        else None,
        "impl_start_epoch": (now - impl_sec)
        if impl_live and impl_sec is not None
        else None,
        "benchmark_start_epoch": (now - bench_sec)
        if bench_live and bench_sec is not None
        else None,
    }


def _find_first_harbor_start(*, artifacts_dir: Path) -> float | None:
    """Find the earliest harbor job start time across all phases."""
    earliest: float | None = None
    for job_dir in _find_all_harbor_job_dirs(artifacts_dir=artifacts_dir):
        agg = _load_aggregate_result(harbor_job_dir=job_dir)
        if agg is None:
            continue
        started = _parse_iso_timestamp(agg.get("started_at"))
        if started is not None and (earliest is None or started < earliest):
            earliest = started

    if earliest is None:
        for phase in ("train_small", "train_remaining", "train_full"):
            mt = _file_mtime(artifacts_dir / phase)
            if mt is not None and (earliest is None or mt < earliest):
                earliest = mt

    return earliest


def _find_latest_summary_mtime(*, artifacts_dir: Path) -> float | None:
    """Find the latest benchmark_summary.json mtime across all phases."""
    latest: float | None = None
    for phase in ("train_small", "train_remaining", "train_full"):
        phase_dir = artifacts_dir / phase
        if not phase_dir.is_dir():
            continue
        mt = _file_mtime(phase_dir / "benchmark_summary.json")
        if mt is not None and (latest is None or mt > latest):
            latest = mt
    return latest


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def _load_manifest(*, run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _load_states(*, run_dir: Path) -> list[AgentState]:
    states_dir = run_dir / "states"
    if not states_dir.is_dir():
        return []
    state_files = sorted(states_dir.glob("iteration-*.json"))
    states: list[AgentState] = []
    for sf in state_files:
        try:
            states.append(AgentState.load(path=sf))
        except (json.JSONDecodeError, ValueError, KeyError, OSError):
            continue

    return states


def _load_single_run(*, run_dir: Path) -> dict[str, Any] | None:
    manifest = _load_manifest(run_dir=run_dir)
    if manifest is None:
        return None

    states = _load_states(run_dir=run_dir)
    benchmark_tasks: list[str] = manifest.get("benchmark_tasks", [])
    train_small_tasks: list[str] = manifest.get("train_small_tasks", [])
    n_samples: int = manifest.get("n_samples", 1)
    failure_inv_model: str = manifest.get("failure_investigation_model", "")
    max_iterations: int = manifest.get("iterations", 25)

    iterations: list[dict[str, Any]] = []
    best_reward: float | None = None

    for state in states:
        status = classify_state_status(state=state, run_root=run_dir)
        parent_iter = parent_iteration_for_state(state=state)
        ps = plan_summary(plan=state.plan)

        result = state.result
        train_small_reward = result.train_small_reward_mean if result else None
        train_full_reward = result.train_full_reward_mean if result else None
        reward_mean = result.reward_mean if result else None
        promoted = train_full_reward is not None

        if result is None or result.reward_mean is None:
            live = _read_live_benchmark_scores(
                artifacts_dir=run_dir
                / "artifacts"
                / f"iteration-{state.iteration:04d}",
            )
            if live["train_small_reward"] is not None:
                train_small_reward = live["train_small_reward"]
            if live["train_full_reward"] is not None:
                train_full_reward = live["train_full_reward"]
            promoted = live["promoted"]

        failure_analyses = [
            {
                "task_name": fa.task_name,
                "general_failure_reason": fa.general_failure_reason,
                "task_specific_explanation": fa.task_specific_explanation,
                "consistency": fa.consistency,
                "suggested_fix_category": fa.suggested_fix_category,
            }
            for fa in state.failure_analyses
        ]

        iter_data: dict[str, Any] = {
            "iteration": state.iteration,
            "status": status,
            "created_at": state.created_at_utc,
            "parent_iteration": parent_iter,
            "plan_summary": ps,
            "completion_rate": None,
            "failure_rate": None,
            "timeout_rate": None,
            "max_iters_rate": None,
            "other_error_rate": None,
            "timeout_or_turn_limit_rate": None,
            "tasks": {},
            "train_small_reward": train_small_reward,
            "train_full_reward": train_full_reward,
            "reward_mean": reward_mean,
            "promoted": promoted,
            "failure_analyses": failure_analyses,
            "n_failure_analyses": len(failure_analyses),
        }

        artifacts_dir = run_dir / "artifacts" / f"iteration-{state.iteration:04d}"

        has_result = result is not None and result.reward_mean is not None
        if has_result:
            metrics = _compute_metrics_from_state(state=state)
            if metrics is not None:
                iter_data.update(metrics)

        if has_result:
            iter_stage = "completed"
        else:
            iter_stage = _detect_current_stage(
                artifacts_dir=artifacts_dir,
                is_bootstrap=(state.iteration == 0),
            )
            if iter_stage == "completed":
                iter_stage = "benchmarking"

        iter_data["status"] = iter_stage
        stage_times = _compute_stage_times(
            artifacts_dir=artifacts_dir,
            is_bootstrap=(state.iteration == 0),
            current_stage=iter_stage,
        )
        iter_data.update(stage_times)

        best_candidate = (
            train_full_reward if train_full_reward is not None else train_small_reward
        )
        if isinstance(best_candidate, float) and (
            best_reward is None or best_candidate > best_reward
        ):
            best_reward = best_candidate

        iterations.append(iter_data)

    if iterations:
        current_iteration = iterations[-1]["iteration"]
        current_stage = iterations[-1]["status"]
    else:
        current_iteration = 0
        current_stage = "waiting"

    current_artifacts = run_dir / "artifacts" / f"iteration-{current_iteration:04d}"
    if current_stage == "completed" and current_iteration < max_iterations:
        next_iter = current_iteration + 1
        next_artifacts = run_dir / "artifacts" / f"iteration-{next_iter:04d}"
        if next_artifacts.exists():
            current_stage = _detect_current_stage(artifacts_dir=next_artifacts)
            if current_stage == "completed":
                next_state_path = run_dir / "states" / f"iteration-{next_iter:04d}.json"
                if next_state_path.exists():
                    try:
                        ns = json.loads(next_state_path.read_text(encoding="utf-8"))
                        if ns.get("result") is None:
                            current_stage = "benchmarking"
                    except (json.JSONDecodeError, OSError):
                        pass
            current_iteration = next_iter
            current_artifacts = next_artifacts
        else:
            current_stage = "planning"

    benchmark_progress = None
    two_tier_progress = None
    if current_stage == "benchmarking":
        benchmark_progress = _get_benchmark_progress(
            artifacts_dir=current_artifacts,
            n_benchmark_tasks=len(benchmark_tasks),
            n_train_small_tasks=len(train_small_tasks),
            n_samples=n_samples,
        )
        two_tier_progress = _get_two_tier_benchmark_progress(
            artifacts_dir=current_artifacts,
        )

    task_aggregate = _build_task_aggregate(
        iterations=iterations,
        benchmark_tasks=benchmark_tasks,
    )

    return {
        "run_dir": run_dir.name,
        "name": manifest.get("name", run_dir.name),
        "model_key": manifest.get("model_key", ""),
        "cursor_model": manifest.get("cursor_model", ""),
        "baseline": manifest.get("baseline", ""),
        "failure_investigation_model": failure_inv_model,
        "n_samples": n_samples,
        "train_small_tasks": train_small_tasks,
        "max_iterations": max_iterations,
        "current_iteration": current_iteration,
        "current_stage": current_stage,
        "benchmark_progress": benchmark_progress,
        "two_tier_progress": two_tier_progress,
        "best_completion": best_reward,
        "iterations": iterations,
        "task_aggregate": task_aggregate,
    }


def _build_task_aggregate(
    *,
    iterations: list[dict[str, Any]],
    benchmark_tasks: list[str],
) -> dict[str, dict[str, Any]]:
    task_outcomes: dict[str, list[str]] = {t: [] for t in benchmark_tasks}

    for it in iterations:
        tasks = it.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_name, outcome in tasks.items():
            if task_name not in task_outcomes:
                task_outcomes[task_name] = []
            task_outcomes[task_name].append(outcome)

    aggregate: dict[str, dict[str, Any]] = {}
    for task_name, outcomes in task_outcomes.items():
        n = len(outcomes) if outcomes else 1
        aggregate[task_name] = {
            "completion_rate": outcomes.count(OUTCOME_PASS) / n if outcomes else 0,
            "failure_rate": outcomes.count(OUTCOME_FAILURE) / n if outcomes else 0,
            "timeout_rate": outcomes.count(OUTCOME_TIMEOUT) / n if outcomes else 0,
            "max_iters_rate": outcomes.count(OUTCOME_MAX_ITERS) / n if outcomes else 0,
            "other_error_rate": (
                outcomes.count(OUTCOME_OTHER_ERROR) / n if outcomes else 0
            ),
            "timeout_or_turn_limit_rate": (
                (outcomes.count(OUTCOME_TIMEOUT) + outcomes.count(OUTCOME_MAX_ITERS))
                / n
                if outcomes
                else 0
            ),
            "n_iterations": len(outcomes),
        }

    return aggregate


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    runs: list[dict[str, Any]] = []
    if OUTPUTS_ROOT.is_dir():
        for child in sorted(OUTPUTS_ROOT.iterdir()):
            if not child.is_dir():
                continue
            run_data = _load_single_run(run_dir=child)
            if run_data is not None:
                runs.append(run_data)

    return JSONResponse(content=runs)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# HTML / JS / CSS
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Evolve V3 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .header { display: flex; justify-content: space-between; align-items: baseline;
            margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  .header-right { font-size: 13px; color: var(--text-muted); }
  .run-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 24px;
  }
  .run-header { display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
  .run-name { font-size: 18px; font-weight: 600; }
  .run-meta { font-size: 13px; color: var(--text-muted); }
  .run-meta span { margin-right: 16px; }
  .progress-section { margin-bottom: 20px; }
  .progress-bar-container {
    background: var(--border); border-radius: 4px; height: 8px;
    margin: 8px 0; overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; background: var(--accent); border-radius: 4px;
    transition: width 0.3s ease;
  }
  .progress-text { font-size: 13px; color: var(--text-muted); display: flex;
                   justify-content: space-between; }
  .stage-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
  }
  .stage-planning { background: #1f2d5c; color: var(--accent); }
  .stage-implementing { background: #2a1f3d; color: var(--purple); }
  .stage-validating { background: #2d2a0f; color: var(--yellow); }
  .stage-benchmarking { background: #2a1a0f; color: var(--orange); }
  .stage-completed { background: #1a2f1a; color: var(--green); }
  .stage-waiting { background: #1a1a1a; color: var(--text-muted); }
  .best-reward { font-size: 28px; font-weight: 700; color: var(--green); }
  .best-reward-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
  .stats-row { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
  .stat-box { text-align: center; }
  .stat-value { font-size: 20px; font-weight: 600; }
  .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .section-title {
    font-size: 14px; font-weight: 600; text-transform: uppercase;
    color: var(--text-muted); margin: 20px 0 12px; letter-spacing: 0.5px;
  }
  .chart-controls { margin-bottom: 12px; }
  .chart-controls select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  }
  .chart-container { position: relative; height: 300px; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--text-muted); font-weight: 600; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.5px; background: var(--surface); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .table-scroll { border: 1px solid var(--border); border-radius: 6px; overflow-x: auto; }
  .cell-pass { color: var(--green); }
  .cell-fail { color: var(--red); }
  .cell-warn { color: var(--yellow); }
  .cell-muted { color: var(--text-muted); }
  .empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
  .benchmark-progress { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
  .tabs { display: flex; gap: 0; margin-bottom: 0; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 8px 16px; font-size: 13px; cursor: pointer; border: none;
    background: none; color: var(--text-muted); border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .run-card.minimized .run-body { display: none; }
  .run-card.minimized { padding: 14px 20px; }
  .run-card.minimized .run-header { margin-bottom: 0; }
  .minimize-btn {
    background: none; border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-muted); cursor: pointer; padding: 2px 10px; font-size: 12px;
    transition: color 0.15s, border-color 0.15s;
  }
  .minimize-btn:hover { color: var(--text); border-color: var(--text-muted); }
  .cell-live { color: var(--accent); }
  .promoted-badge {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
  }
  .promoted-yes { background: #1a2f1a; color: var(--green); }
  .promoted-no { background: #2a1a1a; color: var(--text-muted); }
  .phase-progress {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    font-size: 12px; color: var(--text-muted); margin-top: 6px;
  }
  .phase-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 8px; font-size: 11px;
    border: 1px solid var(--border);
  }
  .phase-chip.done { border-color: var(--green); color: var(--green); }
  .phase-chip.running { border-color: var(--orange); color: var(--orange); }
  .fa-table { margin-top: 8px; }
  .fa-table td { font-size: 12px; vertical-align: top; }
  .fa-reason { font-weight: 600; }
  .fa-explanation { color: var(--text-muted); max-width: 500px; }
  .fa-category {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 10px; border: 1px solid var(--border); color: var(--text-muted);
  }
  .fa-consistency {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 10px;
  }
  .fa-both-fail { background: #2a1a1a; color: var(--red); }
  .fa-different { background: #2d2a0f; color: var(--yellow); }
  .fa-one-pass { background: #1a2f1a; color: var(--green); }
  .iter-select { margin-bottom: 12px; }
  .iter-select select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Agent Evolve V3 Dashboard</h1>
  </div>
  <div class="header-right">
    <span id="last-updated">Loading...</span>
  </div>
</div>
<div id="runs-container">
  <div class="empty-state">Loading runs...</div>
</div>

<script>
const METRIC_LABELS = {
  train_small_reward: 'Train Small Reward',
  train_full_reward: 'Train Full Reward',
  reward_mean: 'Overall Reward',
  failure_rate: 'Failure Rate',
  timeout_or_turn_limit_rate: 'Timeout / Turn Limit Rate',
  timeout_rate: 'Timeout Rate',
  max_iters_rate: 'Max Iters Hit Rate',
  other_error_rate: 'Other Runtime Error Rate',
};
const METRIC_COLORS = {
  train_small_reward: '#58a6ff',
  train_full_reward: '#bc8cff',
  reward_mean: '#3fb950',
  failure_rate: '#f85149',
  timeout_or_turn_limit_rate: '#d29922',
  timeout_rate: '#db6d28',
  max_iters_rate: '#bc8cff',
  other_error_rate: '#8b949e',
};

let charts = {};
let currentMetrics = {};
let minimizedRuns = new Set();
let initializedMinimized = false;
let liveTimerInterval = null;
let selectedFaIter = {};

function fmtDuration(sec) {
  if (sec == null) return '-';
  const s = Math.round(sec);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m + 'm ' + rem + 's';
}

function timingCell(sec, startEpoch) {
  if (startEpoch != null) {
    const elapsed = (Date.now() / 1000) - startEpoch;
    return '<td class="num cell-live" data-stage-start="' + startEpoch + '">' + fmtDuration(elapsed) + '</td>';
  }
  if (sec != null) return '<td class="num">' + fmtDuration(sec) + '</td>';
  return '<td class="num cell-muted">-</td>';
}

function updateLiveTimers() {
  document.querySelectorAll('td[data-stage-start]').forEach(td => {
    const start = parseFloat(td.dataset.stageStart);
    const elapsed = (Date.now() / 1000) - start;
    td.textContent = fmtDuration(elapsed);
  });
}

function toggleMinimize(runDir) {
  if (minimizedRuns.has(runDir)) minimizedRuns.delete(runDir);
  else minimizedRuns.add(runDir);
  if (lastData) renderAll(lastData);
}

function destroyAllCharts() {
  for (const [key, chart] of Object.entries(charts)) chart.destroy();
  charts = {};
}

function stageClass(stage) {
  const map = {planning:'stage-planning', implementing:'stage-implementing',
    validating:'stage-validating', benchmarking:'stage-benchmarking',
    completed:'stage-completed'};
  return map[stage] || 'stage-waiting';
}

function pct(v) {
  if (v == null) return '-';
  const p = v * 100;
  return (p % 1 === 0 ? p.toFixed(0) : p.toFixed(1)) + '%';
}

function fmtReward(v) {
  if (v == null) return '-';
  return v.toFixed(3);
}

function cellClass(val, isGood) {
  if (val == null) return 'cell-muted';
  if (isGood) return val >= 0.7 ? 'cell-pass' : val >= 0.3 ? 'cell-warn' : 'cell-fail';
  return val <= 0.05 ? 'cell-pass' : val <= 0.2 ? 'cell-warn' : 'cell-fail';
}

function rewardClass(val) {
  if (val == null) return 'cell-muted';
  return val >= 0.5 ? 'cell-pass' : val >= 0.25 ? 'cell-warn' : 'cell-fail';
}

function consistencyClass(c) {
  if (c === 'both_same_failure') return 'fa-both-fail';
  if (c === 'different_failures') return 'fa-different';
  if (c === 'one_passed_one_failed') return 'fa-one-pass';
  return '';
}

function consistencyLabel(c) {
  if (c === 'both_same_failure') return 'both fail';
  if (c === 'different_failures') return 'different';
  if (c === 'one_passed_one_failed') return '1 pass / 1 fail';
  return c;
}

function renderPhaseProgress(run) {
  if (!run.two_tier_progress || !run.two_tier_progress.phases.length) return '';
  const phases = run.two_tier_progress.phases;
  let chips = phases.map(p => {
    const label = p.phase.replace('train_', '');
    const cls = p.status === 'done' ? 'done' : 'running';
    const info = p.status === 'done' ? 'done' : p.completed + '/' + p.total;
    return '<span class="phase-chip ' + cls + '">' + label + ': ' + info + '</span>';
  }).join('');
  return '<div class="phase-progress">' + chips + '</div>';
}

function renderFailureAnalyses(run, id) {
  const itersWithFa = run.iterations.filter(i => i.failure_analyses && i.failure_analyses.length > 0);
  if (!itersWithFa.length) return '<div class="cell-muted" style="padding:20px">No failure analyses yet.</div>';

  const selKey = id;
  const selIter = selectedFaIter[selKey] != null ? selectedFaIter[selKey] : itersWithFa[itersWithFa.length - 1].iteration;
  const iter = itersWithFa.find(i => i.iteration === selIter) || itersWithFa[itersWithFa.length - 1];

  let options = itersWithFa.map(i =>
    '<option value="' + i.iteration + '"' + (i.iteration === iter.iteration ? ' selected' : '') + '>Iteration ' + i.iteration + ' (' + i.failure_analyses.length + ' reports)</option>'
  ).join('');

  let rows = '';
  for (const fa of iter.failure_analyses) {
    rows += '<tr>' +
      '<td>' + fa.task_name + '</td>' +
      '<td class="fa-reason">' + fa.general_failure_reason + '</td>' +
      '<td class="fa-explanation">' + fa.task_specific_explanation + '</td>' +
      '<td><span class="fa-consistency ' + consistencyClass(fa.consistency) + '">' + consistencyLabel(fa.consistency) + '</span></td>' +
      '<td><span class="fa-category">' + fa.suggested_fix_category + '</span></td>' +
      '</tr>';
  }

  return '<div class="iter-select"><select onchange="changeFaIter(\\''+id+'\\', parseInt(this.value))">' + options + '</select></div>' +
    '<div class="table-scroll"><table class="fa-table"><thead><tr>' +
    '<th>Task</th><th>Failure Reason</th><th>Explanation</th><th>Consistency</th><th>Fix Category</th>' +
    '</tr></thead><tbody>' + rows + '</tbody></table></div>';
}

function changeFaIter(id, iter) {
  selectedFaIter[id] = iter;
  if (lastData) renderAll(lastData);
}

function renderRun(run, idx) {
  const id = 'run-' + idx;
  const isMin = minimizedRuns.has(run.run_dir);
  const completedIters = run.iterations.filter(i => i.status === 'completed' && (i.train_small_reward != null || i.train_full_reward != null)).length;
  const pctDone = run.max_iterations > 0
    ? (completedIters / run.max_iterations * 100).toFixed(0)
    : 0;

  let benchHtml = '';
  if (run.current_stage === 'benchmarking' && run.benchmark_progress) {
    const bp = run.benchmark_progress;
    benchHtml = '<div class="benchmark-progress">' + bp.completed + '/' + bp.total + ' trials</div>';
  }
  const phaseHtml = run.current_stage === 'benchmarking' ? renderPhaseProgress(run) : '';

  let iterTableRows = '';
  for (const it of run.iterations) {
    const parent = it.parent_iteration != null ? it.parent_iteration : 'root';
    const promotedBadge = it.iteration === 0 ? '<span class="promoted-badge promoted-yes">ROOT</span>'
      : it.promoted ? '<span class="promoted-badge promoted-yes">YES</span>'
      : it.train_small_reward != null ? '<span class="promoted-badge promoted-no">NO</span>'
      : '';
    const faCount = it.n_failure_analyses || 0;
    const faSuffix = faCount > 0 ? ' <span style="font-size:9px;opacity:0.7" title="' + faCount + ' failure analyses">' + faCount + 'fa</span>' : '';
    iterTableRows += '<tr>' +
      '<td class="num">' + it.iteration + '</td>' +
      '<td><span class="stage-badge ' + stageClass(it.status) + '">' + it.status + '</span>' + faSuffix + '</td>' +
      '<td class="num ' + rewardClass(it.train_small_reward) + '">' + fmtReward(it.train_small_reward) + '</td>' +
      '<td class="num ' + rewardClass(it.train_full_reward) + '">' + fmtReward(it.train_full_reward) + '</td>' +
      '<td>' + promotedBadge + '</td>' +
      '<td class="num ' + cellClass(it.failure_rate, false) + '">' + pct(it.failure_rate) + '</td>' +
      '<td class="num ' + cellClass(it.timeout_or_turn_limit_rate, false) + '">' + pct(it.timeout_or_turn_limit_rate) + '</td>' +
      '<td class="num ' + cellClass(it.other_error_rate, false) + '">' + pct(it.other_error_rate) + '</td>' +
      timingCell(it.plan_time_sec, it.plan_start_epoch) +
      timingCell(it.impl_time_sec, it.impl_start_epoch) +
      timingCell(it.benchmark_time_sec, it.benchmark_start_epoch) +
      '<td class="num">' + parent + '</td>' +
      '</tr>';
  }

  const tasks = Object.entries(run.task_aggregate || {}).sort((a,b) => a[0].localeCompare(b[0]));
  let taskRows = '';
  for (const [name, m] of tasks) {
    const inSmall = (run.train_small_tasks || []).includes(name);
    const splitBadge = inSmall ? '<span style="color:var(--accent);font-size:10px"> S</span>' : '';
    taskRows += '<tr><td>' + name + splitBadge + '</td>' +
      '<td class="num ' + cellClass(m.completion_rate, true) + '">' + pct(m.completion_rate) + '</td>' +
      '<td class="num ' + cellClass(m.failure_rate, false) + '">' + pct(m.failure_rate) + '</td>' +
      '<td class="num ' + cellClass(m.timeout_or_turn_limit_rate, false) + '">' + pct(m.timeout_or_turn_limit_rate) + '</td>' +
      '<td class="num ' + cellClass(m.timeout_rate, false) + '">' + pct(m.timeout_rate) + '</td>' +
      '<td class="num ' + cellClass(m.max_iters_rate, false) + '">' + pct(m.max_iters_rate) + '</td>' +
      '<td class="num ' + cellClass(m.other_error_rate, false) + '">' + pct(m.other_error_rate) + '</td>' +
      '<td class="num">' + m.n_iterations + '</td></tr>';
  }

  const nSamples = run.n_samples || 1;
  const fiModel = run.failure_investigation_model || '';
  const metaExtra = nSamples > 1 ? '<span>N=' + nSamples + '</span>' : '';
  const metaFi = fiModel ? '<span>FI: ' + fiModel + '</span>' : '';

  return '' +
  '<div class="run-card' + (isMin ? ' minimized' : '') + '" id="' + id + '" data-run-dir="' + run.run_dir + '">' +
    '<div class="run-header">' +
      '<div style="display:flex;align-items:center;gap:10px">' +
        '<button class="minimize-btn" onclick="toggleMinimize(\\''+run.run_dir+'\\')">'+  (isMin ? '+' : '\\u2212') + '</button>' +
        '<div>' +
          '<div class="run-name">' + run.name + '</div>' +
          '<div class="run-meta">' +
            '<span>Model: ' + run.model_key + '</span>' +
            '<span>Planner: ' + run.cursor_model + '</span>' +
            '<span>Baseline: ' + run.baseline + '</span>' +
            metaExtra + metaFi +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div style="text-align:right">' +
        '<div class="best-reward-label">Best Reward</div>' +
        '<div class="best-reward">' + (run.best_completion != null ? fmtReward(run.best_completion) : '-') + '</div>' +
      '</div>' +
    '</div>' +

    '<div class="run-body">' +
    '<div class="progress-section">' +
      '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">' +
        '<span class="stage-badge ' + stageClass(run.current_stage) + '">' + run.current_stage + '</span>' +
        benchHtml +
      '</div>' +
      phaseHtml +
      '<div class="progress-bar-container">' +
        '<div class="progress-bar-fill" style="width:' + pctDone + '%"></div>' +
      '</div>' +
      '<div class="progress-text">' +
        '<span>Iteration ' + run.current_iteration + ' of ' + run.max_iterations + '</span>' +
        '<span>' + completedIters + ' completed</span>' +
      '</div>' +
    '</div>' +

    '<div class="tabs">' +
      '<button class="tab active" onclick="switchTab(\\''+id+'\\',\\'iterations\\')">Iterations</button>' +
      '<button class="tab" onclick="switchTab(\\''+id+'\\',\\'tasks\\')">Tasks</button>' +
      '<button class="tab" onclick="switchTab(\\''+id+'\\',\\'failures\\')">Failure Analysis</button>' +
      '<button class="tab" onclick="switchTab(\\''+id+'\\',\\'chart\\')">Chart</button>' +
    '</div>' +

    '<div class="tab-content active" data-tab="iterations" data-run="' + id + '">' +
      '<div class="section-title">Iteration History</div>' +
      '<div class="table-scroll">' +
        '<table>' +
          '<thead><tr>' +
            '<th class="num">Iter</th><th>Status</th>' +
            '<th class="num">Small</th><th class="num">Full</th><th>Promoted</th>' +
            '<th class="num">Failure</th><th class="num">Timeout</th><th class="num">Other Err</th>' +
            '<th class="num">Plan</th><th class="num">Impl</th><th class="num">Bench</th><th class="num">Parent</th>' +
          '</tr></thead>' +
          '<tbody>' + iterTableRows + '</tbody>' +
        '</table>' +
      '</div>' +
    '</div>' +

    '<div class="tab-content" data-tab="tasks" data-run="' + id + '">' +
      '<div class="section-title">Per-Task Aggregate <span style="font-size:11px;color:var(--accent)">S = train/small</span></div>' +
      '<div class="table-scroll">' +
        '<table>' +
          '<thead><tr>' +
            '<th>Task</th><th class="num">Completion</th><th class="num">Failure</th><th class="num">Timeout</th>' +
            '<th class="num">Timeout</th><th class="num">Max Iters</th><th class="num">Other Error</th><th class="num">Iters</th>' +
          '</tr></thead>' +
          '<tbody>' + taskRows + '</tbody>' +
        '</table>' +
      '</div>' +
    '</div>' +

    '<div class="tab-content" data-tab="failures" data-run="' + id + '">' +
      '<div class="section-title">Failure Analysis Reports</div>' +
      renderFailureAnalyses(run, id) +
    '</div>' +

    '<div class="tab-content" data-tab="chart" data-run="' + id + '">' +
      '<div class="section-title">Metrics Over Iterations</div>' +
      '<div class="chart-controls">' +
        '<select onchange="changeMetric(\\''+id+'\\', this.value)" id="metric-select-' + id + '">' +
          Object.entries(METRIC_LABELS).map(function(e) {
            return '<option value="' + e[0] + '"' + (e[0] === 'train_small_reward' ? ' selected' : '') + '>' + e[1] + '</option>';
          }).join('') +
        '</select>' +
      '</div>' +
      '<div class="chart-container">' +
        '<canvas id="chart-' + id + '"></canvas>' +
      '</div>' +
    '</div>' +
    '</div>' +
  '</div>';
}

function switchTab(runId, tabName) {
  const card = document.getElementById(runId);
  card.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  card.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const tabs = card.querySelectorAll('.tab');
  const contents = card.querySelectorAll('.tab-content');
  for (let i = 0; i < contents.length; i++) {
    if (contents[i].dataset.tab === tabName) {
      contents[i].classList.add('active');
      tabs[i].classList.add('active');
    }
  }
}

function buildChart(runId, run, metric) {
  const canvas = document.getElementById('chart-' + runId);
  if (!canvas) return;

  if (charts[runId]) {
    charts[runId].destroy();
    delete charts[runId];
  }

  const isRewardMetric = metric === 'train_small_reward' || metric === 'train_full_reward' || metric === 'reward_mean';
  const iters = run.iterations.filter(function(i) {
    return i.status === 'completed' && i[metric] != null;
  });

  const color = METRIC_COLORS[metric] || '#8b949e';
  const label = METRIC_LABELS[metric] || metric;

  charts[runId] = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: label,
        data: iters.map(function(i) { return {x: i.iteration, y: i[metric]}; }),
        backgroundColor: color + '99',
        borderColor: color,
        pointRadius: 6,
        pointHoverRadius: 8,
        showLine: true,
        tension: 0.1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {display: false},
        tooltip: {
          callbacks: {
            label: function(pt) {
              if (isRewardMetric) return 'Iter ' + pt.raw.x + ': ' + pt.raw.y.toFixed(3);
              return 'Iter ' + pt.raw.x + ': ' + (pt.raw.y * 100).toFixed(1) + '%';
            }
          }
        }
      },
      scales: {
        x: {
          title: {display: true, text: 'Iteration', color: '#8b949e'},
          ticks: {color: '#8b949e', stepSize: 1},
          grid: {color: '#30363d44'},
        },
        y: {
          title: {display: true, text: label, color: '#8b949e'},
          ticks: {
            color: '#8b949e',
            callback: function(v) {
              if (isRewardMetric) return v.toFixed(2);
              return (v*100).toFixed(0)+'%';
            }
          },
          grid: {color: '#30363d44'},
          min: 0, max: 1,
        }
      }
    }
  });
}

function changeMetric(runId, metric) {
  currentMetrics[runId] = metric;
  if (lastData) {
    const idx = parseInt(runId.replace('run-',''));
    const run = lastData[idx];
    if (run) buildChart(runId, run, metric);
  }
}

let lastData = null;

function renderAll(runs) {
  const container = document.getElementById('runs-container');

  if (!runs.length) {
    container.innerHTML = '<div class="empty-state">No runs found in outputs/</div>';
    return;
  }

  const sorted = runs.slice().sort(function(a, b) {
    const aMin = minimizedRuns.has(a.run_dir) ? 1 : 0;
    const bMin = minimizedRuns.has(b.run_dir) ? 1 : 0;
    return aMin - bMin;
  });

  const activeTabs = {};
  document.querySelectorAll('.run-card').forEach(function(card) {
    const activeContent = card.querySelector('.tab-content.active');
    if (activeContent) activeTabs[card.dataset.runDir] = activeContent.dataset.tab;
  });

  destroyAllCharts();
  container.innerHTML = sorted.map(function(r, i) { return renderRun(r, i); }).join('');

  document.querySelectorAll('.run-card').forEach(function(card) {
    const dir = card.dataset.runDir;
    if (dir && activeTabs[dir]) switchTab(card.id, activeTabs[dir]);
  });

  sorted.forEach(function(run, i) {
    if (minimizedRuns.has(run.run_dir)) return;
    var id = 'run-' + i;
    var metric = currentMetrics[id] || 'train_small_reward';
    var select = document.getElementById('metric-select-' + id);
    if (select) select.value = metric;
    buildChart(id, run, metric);
  });

  if (liveTimerInterval) clearInterval(liveTimerInterval);
  if (document.querySelectorAll('td[data-stage-start]').length > 0) {
    liveTimerInterval = setInterval(updateLiveTimers, 1000);
  }
}

async function fetchRuns() {
  try {
    const resp = await fetch('/api/runs');
    const runs = await resp.json();
    lastData = runs;
    if (!initializedMinimized) {
      initializedMinimized = true;
      runs.forEach(function(r) { minimizedRuns.add(r.run_dir); });
    }
    renderAll(runs);
    document.getElementById('last-updated').textContent =
      'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

fetchRuns();
setInterval(fetchRuns, 5000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="agent_evolve_v3.dashboard:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
