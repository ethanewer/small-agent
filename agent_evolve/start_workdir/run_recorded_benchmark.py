from __future__ import annotations

import argparse
import contextlib
import filecmp
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime  # type: ignore
from pathlib import Path
from typing import Iterator, cast

SNAPSHOT_IGNORES = shutil.ignore_patterns(
    "*.log",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".basedpyright",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Harbor benchmark and record code+eval artifacts.",
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("harbor/run_debug.sh"),
        help="Benchmark runner path relative to repo root.",
    )
    parser.add_argument(
        "--agent-key",
        type=str,
        default="terminus-2",
        help="Agent key passed to Harbor runner.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Optional model key override passed to Harbor runner.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="run",
        help="Label prefix for artifact directories (e.g. 'run' or 'eval').",
    )
    parser.add_argument(
        "--runner-args",
        type=str,
        default="",
        help='Extra arguments forwarded to the runner script (e.g. "--split 1").',
    )
    return parser.parse_args(argv)


def _iter_root(*, workdir_root: Path, iteration: int) -> tuple[Path, Path]:
    iter_name = f"iter-{iteration:04d}"
    run_root = workdir_root.parent
    snapshots_iter_dir = run_root / "snapshots" / iter_name
    evals_iter_dir = run_root / "eval" / iter_name
    snapshots_iter_dir.mkdir(parents=True, exist_ok=True)
    evals_iter_dir.mkdir(parents=True, exist_ok=True)
    return snapshots_iter_dir, evals_iter_dir


def _next_run_dir(
    *, iter_dir: Path, label: str = "run", create_dir: bool = True
) -> Path:
    existing = sorted(path for path in iter_dir.glob(f"{label}-*") if path.is_dir())
    next_index = len(existing) + 1
    run_dir = iter_dir / f"{label}-{next_index:04d}"
    if create_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _copy_code_snapshot(*, workdir_root: Path, target_dir: Path) -> None:
    shutil.copytree(
        src=workdir_root,
        dst=target_dir,
        ignore=SNAPSHOT_IGNORES,
        dirs_exist_ok=False,
    )


def _collect_job_dirs(*, jobs_root: Path) -> set[str]:
    if not jobs_root.exists():
        return set()

    return {path.name for path in jobs_root.iterdir() if path.is_dir()}


def _resolve_new_job_dir(*, jobs_root: Path, before: set[str]) -> Path:
    after = _collect_job_dirs(jobs_root=jobs_root)
    new_names = sorted(after - before)
    if not new_names:
        raise RuntimeError("Unable to determine newly created Harbor jobs directory.")

    return jobs_root / new_names[-1]


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _benchmark_lock(*, jobs_root: Path) -> Iterator[None]:
    jobs_root.mkdir(parents=True, exist_ok=True)
    lock_path = jobs_root / ".agent_evolve_benchmark.lock"
    my_pid = str(os.getpid())

    if lock_path.exists():
        try:
            stored_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            stored_pid = -1

        if stored_pid > 0 and _is_pid_alive(stored_pid):
            raise RuntimeError(
                f"Another benchmark run (PID {stored_pid}) is active. "
                f"Lock file: {lock_path}"
            )
        lock_path.unlink(missing_ok=True)

    lock_path.write_text(my_pid, encoding="utf-8")

    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


@contextlib.contextmanager
def _swap_agent(*, workspace_agent: Path, deployed_agent: Path) -> Iterator[None]:
    """Temporarily replace the deployed core_agent.py with the workspace agent.py."""
    if not workspace_agent.exists():
        yield
        return

    backup = deployed_agent.with_suffix(".py.bak")
    shutil.copy2(src=deployed_agent, dst=backup)
    try:
        shutil.copy2(src=workspace_agent, dst=deployed_agent)
        if not filecmp.cmp(workspace_agent, deployed_agent, shallow=False):
            raise RuntimeError(
                "Agent swap failed: deployed file does not match workspace agent.py"
            )
        print(f"[agent-swap] deployed workspace agent.py -> {deployed_agent}")
        yield
    finally:
        shutil.copy2(src=backup, dst=deployed_agent)
        backup.unlink(missing_ok=True)
        print(f"[agent-swap] restored original -> {deployed_agent}")


def _load_run_summary(*, harbor_job_dir: Path) -> dict[str, object]:
    for result_path in sorted(harbor_job_dir.glob("*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "reward_stats" in data:
            return cast(dict[str, object], data)

    return {}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    # This script lives at <run_root>/agent_evolve/run_recorded_benchmark.py.
    # Artifacts should be written under that run root (not repository-level outputs).
    workdir_root = script_path.parent
    repo_root: Path | None = None
    for candidate in script_path.parents:
        if (candidate / "harbor" / "run_small_benchmark.sh").exists():
            repo_root = candidate
            break
    if repo_root is None:
        raise RuntimeError("Unable to locate repository root from script path.")

    jobs_root = repo_root / "harbor" / "jobs"
    snapshots_iter_dir, evals_iter_dir = _iter_root(
        workdir_root=workdir_root,
        iteration=args.iteration,
    )

    label = args.run_label
    snapshot_run_dir = _next_run_dir(
        iter_dir=snapshots_iter_dir, label=label, create_dir=False
    )
    eval_run_dir = _next_run_dir(iter_dir=evals_iter_dir, label=label)
    _copy_code_snapshot(workdir_root=workdir_root, target_dir=snapshot_run_dir)

    runner_path = (repo_root / args.runner).resolve()
    extra_args = shlex.split(args.runner_args) if args.runner_args else []
    command: list[str] = [str(runner_path), *extra_args, "--agent", args.agent_key]
    if args.model_key:
        command.extend(["--model", args.model_key])

    workspace_agent = workdir_root / "agent.py"
    deployed_agent = repo_root / "agents" / "terminus2" / "core_agent.py"

    with (
        _benchmark_lock(jobs_root=jobs_root),
        _swap_agent(
            workspace_agent=workspace_agent,
            deployed_agent=deployed_agent,
        ),
    ):
        before_jobs = _collect_job_dirs(jobs_root=jobs_root)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        (eval_run_dir / "benchmark_stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (eval_run_dir / "benchmark_stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        raise SystemExit(completed.returncode)

    harbor_job_dir = _resolve_new_job_dir(
        jobs_root=jobs_root,
        before=before_jobs,
    )
    target_harbor_dir = eval_run_dir / "harbor_job"
    shutil.copytree(src=harbor_job_dir, dst=target_harbor_dir, dirs_exist_ok=False)
    (eval_run_dir / "benchmark_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (eval_run_dir / "benchmark_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )

    run_summary = _load_run_summary(harbor_job_dir=harbor_job_dir)
    reward_stats = run_summary.get("reward_stats")
    reward_mean: float | None = None
    if isinstance(reward_stats, dict):
        candidate = reward_stats.get("mean")
        if isinstance(candidate, int | float):  # pyright: ignore[reportGeneralTypeIssues]
            reward_mean = float(candidate)

    eval_summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "iteration": args.iteration,
        "run_label": label,
        "snapshot_path": str(snapshot_run_dir),
        "eval_path": str(eval_run_dir),
        "runner": str(args.runner),
        "agent_key": args.agent_key,
        "model_key": args.model_key,
        "harbor_job_dir": str(target_harbor_dir),
        "harbor_run_id": harbor_job_dir.name,
        "reward_mean": reward_mean,
        "n_trials": run_summary.get("n_trials"),
        "n_evals": run_summary.get("n_evals"),
    }
    summary_json = json.dumps(eval_summary, indent=2, ensure_ascii=True) + "\n"
    for path in (
        eval_run_dir / "eval_summary.json",
        evals_iter_dir.parent / f"latest_{label}.json",
    ):
        path.write_text(summary_json, encoding="utf-8")

    print(json.dumps(eval_summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
