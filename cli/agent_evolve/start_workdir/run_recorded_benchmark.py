from __future__ import annotations

import argparse
import contextlib
import json
from datetime import UTC, datetime  # type: ignore
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
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
        default=Path("cli/harbor/run_small.sh"),
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
    return parser.parse_args(argv)


def _iter_root(*, workdir_root: Path, iteration: int) -> tuple[Path, Path]:
    iter_name = f"iter-{iteration:04d}"
    run_root = workdir_root.parent
    snapshots_iter_dir = run_root / "snapshots" / iter_name
    evals_iter_dir = run_root / "eval" / iter_name
    snapshots_iter_dir.mkdir(parents=True, exist_ok=True)
    evals_iter_dir.mkdir(parents=True, exist_ok=True)
    return snapshots_iter_dir, evals_iter_dir


def _next_run_dir(*, iter_dir: Path, create_dir: bool = True) -> Path:
    existing = sorted(path for path in iter_dir.glob("run-*") if path.is_dir())
    next_index = len(existing) + 1
    run_dir = iter_dir / f"run-{next_index:04d}"
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


def _is_dated_job_dir_name(name: str) -> bool:
    return bool(re.search(r"\d{4}[-_]?\d{2}[-_]?\d{2}", name))


def _resolve_new_job_dir(
    *, jobs_root: Path, before: set[str], started_at: datetime
) -> Path:
    after = _collect_job_dirs(jobs_root=jobs_root)
    created_paths = sorted(
        (jobs_root / name for name in (after - before)), key=lambda path: path.name
    )
    if not created_paths:
        raise RuntimeError("Unable to determine newly created Harbor jobs directory.")

    dated_created_paths = [
        path for path in created_paths if _is_dated_job_dir_name(path.name)
    ]
    if dated_created_paths:
        return dated_created_paths[-1]

    started_epoch = started_at.timestamp()
    all_job_paths = sorted(
        (path for path in jobs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    recent_paths = [
        path for path in all_job_paths if path.stat().st_mtime >= started_epoch - 1.0
    ]
    recent_dated_paths = [
        path for path in recent_paths if _is_dated_job_dir_name(path.name)
    ]
    if recent_dated_paths:
        return recent_dated_paths[-1]

    return created_paths[-1]


@contextlib.contextmanager
def _benchmark_lock(*, jobs_root: Path) -> Iterator[None]:
    jobs_root.mkdir(parents=True, exist_ok=True)
    lock_path = jobs_root / ".agent_evolve_benchmark.lock"
    try:
        fd = lock_path.open(mode="x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another benchmark run appears to be active. Lock file already exists: {lock_path}"
        ) from exc

    try:
        fd.write(f"pid={os.getpid()} started_at={datetime.now(UTC).isoformat()}\n")
        fd.flush()
        yield
    finally:
        fd.close()
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


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
        if (candidate / "cli" / "harbor" / "run_small.sh").exists():
            repo_root = candidate
            break
    if repo_root is None:
        raise RuntimeError("Unable to locate repository root from script path.")

    cli_root = repo_root / "cli"
    jobs_root = cli_root / "harbor" / "jobs"
    snapshots_iter_dir, evals_iter_dir = _iter_root(
        workdir_root=workdir_root,
        iteration=args.iteration,
    )

    snapshot_run_dir = _next_run_dir(iter_dir=snapshots_iter_dir, create_dir=False)
    eval_run_dir = _next_run_dir(iter_dir=evals_iter_dir)
    _copy_code_snapshot(workdir_root=workdir_root, target_dir=snapshot_run_dir)

    runner_path = (repo_root / args.runner).resolve()
    command: list[str] = [str(runner_path), "--agent", args.agent_key]
    if args.model_key:
        command.extend(["--model", args.model_key])

    benchmark_started_at = datetime.now(UTC)
    with _benchmark_lock(jobs_root=jobs_root):
        before_jobs = _collect_job_dirs(jobs_root=jobs_root)
        completed = subprocess.run(
            command,
            cwd=cli_root,
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
        started_at=benchmark_started_at,
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
    eval_summary_path = eval_run_dir / "eval_summary.json"
    eval_summary_path.write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    latest_eval_path = evals_iter_dir.parent / "latest_eval.json"
    latest_eval_path.write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(eval_summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
