from __future__ import annotations

import argparse
import json
import shutil
import signal
from dataclasses import dataclass
from datetime import UTC, datetime  # pyright: ignore[reportAttributeAccessIssue]
from pathlib import Path
import subprocess
import sys
from typing import cast


@dataclass
class StopState:
    requested: bool = False


COPY_IGNORES = shutil.ignore_patterns(
    "outputs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".basedpyright",
    "*.log",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run outer improve/eval loop for evolver workdir.",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("cli/harbor/run_small.sh"),
    )
    parser.add_argument("--agent-key", type=str, default="terminus-2")
    parser.add_argument("--model-key", type=str, default=None)
    parser.add_argument("--cursor-model", type=str, default=None)
    return parser.parse_args(argv)


def _create_run_root(*, outputs_root: Path) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    candidate = outputs_root / run_id
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = outputs_root / f"{run_id}-{suffix:02d}"

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _seed_run_workdir(*, template_root: Path, run_workdir: Path) -> None:
    shutil.copytree(
        src=template_root,
        dst=run_workdir,
        ignore=COPY_IGNORES,
        dirs_exist_ok=False,
    )


def _load_state(*, state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return {}

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return cast(dict[str, object], payload)

    return {}


def _save_state(*, state_path: Path, payload: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _latest_eval_for_iteration(*, workdir_root: Path, iteration: int) -> Path | None:
    iter_dir = workdir_root / "eval" / f"iter-{iteration:04d}"
    if not iter_dir.exists():
        return None

    run_dirs = sorted(path for path in iter_dir.glob("run-*") if path.is_dir())
    if not run_dirs:
        return None

    return run_dirs[-1] / "eval_summary.json"


def _render_prompt(
    *,
    template_path: Path,
    iteration: int,
    run_root: Path,
    workdir_root: Path,
    eval_root: Path,
    snapshot_root: Path,
    eval_summary_path: Path,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    eval_artifacts_path = eval_summary_path.parent
    return text.format(
        iteration=iteration,
        iteration_padded=f"{iteration:04d}",
        run_root=run_root,
        workdir_root=workdir_root,
        eval_root=eval_root,
        snapshot_root=snapshot_root,
        eval_summary_path=eval_summary_path,
        eval_artifacts_path=eval_artifacts_path,
    )


def _run_command(*, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _record_step_output(
    *,
    target_path: Path,
    completed: subprocess.CompletedProcess[str],
) -> None:
    payload = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent
    template_root = root_dir / "start_workdir"
    outputs_root = root_dir / "outputs"
    run_root = _create_run_root(outputs_root=outputs_root)
    workdir_root = run_root / "agent_evolve"
    eval_root = run_root / "eval"
    snapshot_root = run_root / "snapshots"
    eval_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    _seed_run_workdir(template_root=template_root, run_workdir=workdir_root)
    template_path = root_dir / "headless_inner_loop_prompt.md"
    state_path = run_root / "run_state.json"

    stop_state = StopState()

    def _request_stop(signum: int, _frame: object) -> None:
        del _frame
        stop_state.requested = True
        print(f"Received signal {signum}. Will stop after current step.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    state = _load_state(state_path=state_path)
    current_iteration = max(int(state.get("next_iteration", args.start_iteration)), 1)  # pyright: ignore[reportArgumentType]
    final_iteration = max(args.iterations, current_iteration)

    while current_iteration <= final_iteration:
        if stop_state.requested:
            break

        print(f"=== outer iteration {current_iteration}/{final_iteration} ===")
        iteration_dir = eval_root / f"iter-{current_iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        benchmark_command = [
            "uv",
            "run",
            "python",
            "agent_evolve/run_recorded_benchmark.py",
            "--iteration",
            str(current_iteration),
            "--runner",
            str(args.runner),
            "--agent-key",
            args.agent_key,
        ]
        if args.model_key:
            benchmark_command.extend(["--model-key", args.model_key])
        benchmark_step = _run_command(command=benchmark_command, cwd=run_root)
        _record_step_output(
            target_path=iteration_dir / "outer_benchmark_step.json",
            completed=benchmark_step,
        )
        if benchmark_step.returncode != 0:
            print(benchmark_step.stdout)
            print(benchmark_step.stderr, file=sys.stderr)
            return benchmark_step.returncode

        eval_summary_path = _latest_eval_for_iteration(
            workdir_root=run_root,
            iteration=current_iteration,
        )
        if eval_summary_path is None:
            raise RuntimeError("Expected eval summary after benchmark run.")

        prompt_text = _render_prompt(
            template_path=template_path,
            iteration=current_iteration,
            run_root=run_root,
            workdir_root=workdir_root,
            eval_root=eval_root,
            snapshot_root=snapshot_root,
            eval_summary_path=eval_summary_path,
        )
        prompt_file = iteration_dir / "cursor_prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        cursor_command = [
            "agent",
            "--print",
            "--force",
            "--trust",
            "--sandbox",
            "enabled",
            "--workspace",
            str(run_root),
        ]
        if args.cursor_model:
            cursor_command.extend(["--model", args.cursor_model])
        cursor_command.append(prompt_text)
        cursor_step = _run_command(command=cursor_command, cwd=run_root)
        _record_step_output(
            target_path=iteration_dir / "outer_cursor_step.json",
            completed=cursor_step,
        )
        if cursor_step.returncode != 0:
            print(cursor_step.stdout)
            print(cursor_step.stderr, file=sys.stderr)
            return cursor_step.returncode

        test_command = [
            "uv",
            "run",
            "python",
            "-m",
            "unittest",
            "agent_evolve.test_interface_contract",
        ]
        test_step = _run_command(command=test_command, cwd=run_root)
        _record_step_output(
            target_path=iteration_dir / "outer_validation_step.json",
            completed=test_step,
        )
        if test_step.returncode != 0:
            print(test_step.stdout)
            print(test_step.stderr, file=sys.stderr)
            return test_step.returncode

        state = {
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "last_completed_iteration": current_iteration,
            "next_iteration": current_iteration + 1,
            "stop_requested": stop_state.requested,
            "runner": str(args.runner),
            "agent_key": args.agent_key,
            "model_key": args.model_key,
            "run_root": str(run_root),
        }
        _save_state(state_path=state_path, payload=state)
        current_iteration += 1

    state = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "last_completed_iteration": current_iteration - 1,
        "next_iteration": current_iteration,
        "stop_requested": stop_state.requested,
        "runner": str(args.runner),
        "agent_key": args.agent_key,
        "model_key": args.model_key,
        "run_root": str(run_root),
        "stopped_gracefully": stop_state.requested,
    }
    _save_state(state_path=state_path, payload=state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
