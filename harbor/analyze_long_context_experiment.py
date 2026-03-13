#!/usr/bin/env python3
"""Analyze long context experiment Slurm logs and result.json files."""

from __future__ import annotations

import json
import re
from pathlib import Path

LOG_DIR = (
    Path(__file__).parent
    / "jobs/long_context_experiment_2026-03-12__05-46-04/slurm_logs"
)
JOBS_ROOT = Path(__file__).parent / "jobs"


def parse_log_file(path: Path) -> dict | None:
    """Extract jobs_dir, trials, errors, mean, exception distribution from a log file."""
    content = path.read_text()
    # jobs_dir=... (path may have spaces, take until next space or end)
    jobs_match = re.search(r"jobs_dir=(\S+)", content)
    if not jobs_match:
        return None
    jobs_dir = jobs_match.group(1).strip()

    # Trials | 17
    trials_match = re.search(r"│ Trials\s+│\s+(\d+)", content)
    trials = int(trials_match.group(1)) if trials_match else 0

    # Errors | 13
    errors_match = re.search(r"│ Errors\s+│\s+(\d+)", content)
    errors = int(errors_match.group(1)) if errors_match else 0

    # Mean | 0.050
    mean_match = re.search(r"│ Mean\s+│\s+([\d.]+)", content)
    mean = float(mean_match.group(1)) if mean_match else 0.0

    # Exception Distribution - RuntimeError | 3, AgentTimeoutError | 10
    exc_dist: dict[str, int] = {}
    for exc_match in re.finditer(
        r"│\s+(RuntimeError|AgentTimeoutError)\s+│\s+(\d+)", content
    ):
        exc_dist[exc_match.group(1)] = int(exc_match.group(2))

    # reward = 1.0 count for Passes
    reward_1_match = re.search(r"reward = 1\.0\s+│\s+(\d+)", content)
    passes = int(reward_1_match.group(1)) if reward_1_match else 0

    return {
        "model": path.stem.rsplit("_", 1)[0],
        "jobs_dir": jobs_dir,
        "trials": trials,
        "errors": errors,
        "mean": mean,
        "runtime_errors": exc_dist.get("RuntimeError", 0),
        "agent_timeout_errors": exc_dist.get("AgentTimeoutError", 0),
        "passes": passes,
    }


def get_errored_tasks(jobs_dir: str) -> list[tuple[str, str, str | None]]:
    """Get (task_name, exception_type, brief_message) for each errored trial."""
    jobs_path = Path(jobs_dir)
    if not jobs_path.exists():
        return []

    # Run-level result.json is at jobs_dir/{date_stamp}/result.json
    run_result = None
    run_subdir: Path | None = None
    for d in jobs_path.iterdir():
        if d.is_dir():
            rj = d / "result.json"
            if rj.exists():
                try:
                    data = json.loads(rj.read_text())
                    if "stats" in data and "evals" in data.get("stats", {}):
                        run_result = data
                        run_subdir = d
                        break
                except (json.JSONDecodeError, OSError):
                    pass

    if not run_result or not run_subdir:
        return []

    result: list[tuple[str, str, str | None]] = []
    evals = run_result.get("stats", {}).get("evals", {})
    for eval_data in evals.values():
        exc_stats = eval_data.get("exception_stats", {})
        for exc_type, trial_ids in exc_stats.items():
            for trial_id in trial_ids:
                task_name = (
                    trial_id.rsplit("__", 1)[0] if "__" in trial_id else trial_id
                )
                result.append((task_name, exc_type, None))

    # Get exception_message from per-trial result.json
    for trial_dir in run_subdir.iterdir():
        if trial_dir.is_dir() and "__" in trial_dir.name:
            rj = trial_dir / "result.json"
            if rj.exists():
                try:
                    data = json.loads(rj.read_text())
                    exc_info = data.get("exception_info")
                    if exc_info:
                        task_name = trial_dir.name.rsplit("__", 1)[0]
                        exc_type = exc_info.get("exception_type", "")
                        msg = exc_info.get("exception_message", "")
                        if msg and "Harbor setup" in msg:
                            brief = "Harbor setup/bootstrap failure"
                        elif msg and "mteb" in msg.lower():
                            brief = "mteb-leaderboard Docker pull (image too large)"
                        elif msg and "timeout" in msg.lower():
                            brief = "Agent timeout"
                        else:
                            brief = (msg[:80] + "...") if len(msg) > 80 else msg
                        for i, (t, et, m) in enumerate(result):
                            if t == task_name and et == exc_type and m is None:
                                result[i] = (t, et, brief)
                                break
                except (json.JSONDecodeError, OSError):
                    pass

    return result


def main() -> None:
    log_files = sorted(LOG_DIR.glob("*.out"))
    if not log_files:
        print(f"No .out files in {LOG_DIR}")
        return

    rows: list[dict] = []
    model_errors: dict[str, list[tuple[str, str, str | None]]] = {}
    task_to_models: dict[str, set[str]] = {}  # task -> set of models that errored on it

    for log_path in log_files:
        parsed = parse_log_file(log_path)
        if not parsed:
            continue
        rows.append(parsed)
        if parsed["errors"] > 0:
            errored_tasks = get_errored_tasks(parsed["jobs_dir"])
            model_errors[parsed["model"]] = errored_tasks
            for task_name, _, _ in errored_tasks:
                task_to_models.setdefault(task_name, set()).add(parsed["model"])

    # A) Summary table (sort by model name for consistency)
    rows_sorted = sorted(rows, key=lambda x: (x["model"].split("-")[1], x["model"]))
    print("=" * 90)
    print("A) SUMMARY TABLE")
    print("=" * 90)
    print(
        f"{'Model':<25} {'Trials':>7} {'Errors':>7} {'Mean':>8} "
        f"{'RuntimeErr':>11} {'AgentTimeout':>13} {'Passes':>7}"
    )
    print("-" * 90)
    for r in rows_sorted:
        print(
            f"{r['model']:<25} {r['trials']:>7} {r['errors']:>7} {r['mean']:>8.3f} "
            f"{r['runtime_errors']:>11} {r['agent_timeout_errors']:>13} {r['passes']:>7}"
        )
    print()

    # B) Per-model errored tasks
    print("=" * 90)
    print("B) ERRORED TASKS BY MODEL")
    print("=" * 90)
    for model in sorted(model_errors.keys()):
        tasks = model_errors[model]
        # Deduplicate by (task_name, exception_type)
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str, str | None]] = []
        for t, et, m in tasks:
            key = (t, et)
            if key not in seen:
                seen.add(key)
                unique.append((t, et, m))
        print(f"\n{model}:")
        for task_name, exc_type, msg in sorted(unique, key=lambda x: (x[0], x[1])):
            msg_str = f" ({msg})" if msg else ""
            print(f"  - {task_name}: {exc_type}{msg_str}")
    print()

    # C) Tasks that error on ALL models
    all_models = {r["model"] for r in rows}
    universal_failures = [
        task for task, models in task_to_models.items() if models == all_models
    ]
    print("=" * 90)
    print("C) TASKS THAT ERROR ON ALL MODELS (likely infrastructure)")
    print("=" * 90)
    if universal_failures:
        for t in sorted(universal_failures):
            print(f"  - {t}")
    else:
        print("  (none)")
    print()

    # D) Tasks that error on SOME models only (worth retrying)
    partial_failures = [
        task for task, models in task_to_models.items() if models != all_models
    ]
    print("=" * 90)
    print("D) TASKS THAT ERROR ON SOME MODELS ONLY (worth retrying)")
    print("=" * 90)
    if partial_failures:
        for t in sorted(partial_failures):
            models = sorted(task_to_models[t])
            print(f"  - {t}: fails on {len(models)}/15 models: {', '.join(models)}")
    else:
        print("  (none)")
    print()


if __name__ == "__main__":
    main()
