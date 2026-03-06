from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
from dataclasses import dataclass
from datetime import UTC, datetime  # pyright: ignore[reportAttributeAccessIssue]
from pathlib import Path
import subprocess
import sys
from typing import Mapping, cast


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
    parser.add_argument(
        "--eval-runner",
        type=Path,
        default=Path("cli/harbor/run_eval.sh"),
        help="Runner script for the held-out eval benchmark (run after each iteration).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing run directory to resume from.",
    )
    parser.add_argument(
        "--skip-initial-benchmark",
        action="store_true",
        default=False,
        help="Skip the dev benchmark on iteration 1 if a cached baseline exists.",
    )
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


def _save_state(*, state_path: Path, payload: Mapping[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _latest_eval_for_iteration(
    *, workdir_root: Path, iteration: int, label: str = "run"
) -> Path | None:
    iter_dir = workdir_root / "eval" / f"iter-{iteration:04d}"
    if not iter_dir.exists():
        return None

    run_dirs = sorted(path for path in iter_dir.glob(f"{label}-*") if path.is_dir())
    if not run_dirs:
        return None

    return run_dirs[-1] / "eval_summary.json"


def _resolve_context_length(*, config_path: Path, model_key: str | None) -> int:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        models = data.get("models", {})
        key = model_key or data.get("default_model", "")
        model_cfg = models.get(key, {})
        ctx = model_cfg.get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            return ctx
    except (json.JSONDecodeError, OSError):
        pass

    return 0


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_eval_summary_fields(
    eval_summary_path: Path,
) -> tuple[str, str]:
    """Return (dev_score, dev_trials) as display strings."""
    try:
        data = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        reward_mean = data.get("reward_mean")
        n_trials = data.get("n_trials")
        score_str = (
            f"{reward_mean:.2f}" if isinstance(reward_mean, (int, float)) else "N/A"
        )
        trials_str = str(n_trials) if n_trials is not None else "N/A"
        return score_str, trials_str
    except (json.JSONDecodeError, OSError):
        return "N/A", "N/A"


def _render_prompt(
    *,
    template_path: Path,
    iteration: int,
    run_root: Path,
    workdir_root: Path,
    eval_root: Path,
    snapshot_root: Path,
    eval_summary_path: Path,
    context_length: int,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    eval_artifacts_path = eval_summary_path.parent
    dev_score, dev_trials = _read_eval_summary_fields(eval_summary_path)
    return text.format(
        iteration=iteration,
        iteration_padded=f"{iteration:04d}",
        run_root=run_root,
        workdir_root=workdir_root,
        eval_root=eval_root,
        snapshot_root=snapshot_root,
        eval_summary_path=eval_summary_path,
        eval_artifacts_path=eval_artifacts_path,
        context_length=context_length,
        dev_score=dev_score,
        dev_trials=dev_trials,
    )


def _run_command(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
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


def _is_transient_cursor_error(*, completed: subprocess.CompletedProcess[str]) -> bool:
    if completed.returncode == 0:
        return False

    error_text = f"{completed.stdout}\n{completed.stderr}".lower()
    transient_signals = (
        "http/2 stream closed",
        "error code cancel",
        "stream closed with error code cancel",
        "connection reset",
    )
    return any(token in error_text for token in transient_signals)


STEP_ORDER = ("dev_benchmark", "cursor", "validation", "eval")


def _build_state(
    *,
    stop_state: StopState,
    args: argparse.Namespace,
    run_root: Path,
    current_iteration: int,
    last_completed_step: str | None,
    eval_score: float | None = None,
    last_eval_agent_hash: str | None = None,
    dev_benchmark_failed: bool = False,
    eval_benchmark_failed: bool = False,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "current_iteration": current_iteration,
        "last_completed_step": last_completed_step,
        "stop_requested": stop_state.requested,
        "runner": str(args.runner),
        "eval_runner": str(args.eval_runner),
        "agent_key": args.agent_key,
        "model_key": args.model_key,
        "run_root": str(run_root),
        "eval_score": eval_score,
        "last_eval_agent_hash": last_eval_agent_hash,
        "dev_benchmark_failed": dev_benchmark_failed,
        "eval_benchmark_failed": eval_benchmark_failed,
    }
    if extra:
        state.update(extra)
    return state


def _should_skip_step(step: str, last_completed_step: str | None) -> bool:
    """Return True if *step* was already completed in a prior run of this iteration."""
    if last_completed_step is None:
        return False
    try:
        completed_idx = STEP_ORDER.index(last_completed_step)
        step_idx = STEP_ORDER.index(step)
    except ValueError:
        return False

    return step_idx <= completed_idx


def main(argv: list[str]) -> int:  # noqa: C901, PLR0912, PLR0915
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent
    repo_root = root_dir.parent.parent
    template_root = root_dir / "start_workdir"
    outputs_root = root_dir / "outputs"

    if args.resume:
        run_root = Path(args.resume).resolve()
        if not (run_root / "run_state.json").exists():
            print(f"No run_state.json found in {run_root}", file=sys.stderr)
            return 1
        print(f"Resuming run from {run_root}")
    else:
        run_root = _create_run_root(outputs_root=outputs_root)
        workdir_root_init = run_root / "agent_evolve"
        _seed_run_workdir(template_root=template_root, run_workdir=workdir_root_init)

    workdir_root = run_root / "agent_evolve"
    eval_root = run_root / "eval"
    snapshot_root = run_root / "snapshots"
    eval_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    template_path = root_dir / "headless_inner_loop_prompt.md"
    state_path = run_root / "run_state.json"
    config_path = repo_root / "cli" / "config.json"
    context_length = _resolve_context_length(
        config_path=config_path, model_key=args.model_key
    )

    stop_state = StopState()

    def _request_stop(signum: int, _frame: object) -> None:
        del _frame
        stop_state.requested = True
        print(f"Received signal {signum}. Will stop after current step.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    state = _load_state(state_path=state_path)
    _raw_iter = state.get("current_iteration", args.start_iteration)
    current_iteration = max(int(_raw_iter), 1)  # pyright: ignore[reportArgumentType]
    last_completed_step = cast(str | None, state.get("last_completed_step"))
    if last_completed_step == "eval":
        current_iteration += 1
        last_completed_step = None

    final_iteration = max(args.iterations, current_iteration)
    eval_score: float | None = None
    if isinstance(state.get("eval_score"), (int, float)):
        eval_score = float(state["eval_score"])  # pyright: ignore[reportArgumentType]
    last_eval_agent_hash: str | None = state.get("last_eval_agent_hash")  # pyright: ignore[reportAssignmentType]

    def _save(
        step: str | None,
        *,
        dev_benchmark_failed: bool = False,
        eval_benchmark_failed: bool = False,
    ) -> None:
        nonlocal last_completed_step
        last_completed_step = step
        payload = _build_state(
            stop_state=stop_state,
            args=args,
            run_root=run_root,
            current_iteration=current_iteration,
            last_completed_step=step,
            eval_score=eval_score,
            last_eval_agent_hash=last_eval_agent_hash,
            dev_benchmark_failed=dev_benchmark_failed,
            eval_benchmark_failed=eval_benchmark_failed,
        )
        _save_state(state_path=state_path, payload=payload)

    while current_iteration <= final_iteration:
        if stop_state.requested:
            break

        print(f"=== outer iteration {current_iteration}/{final_iteration} ===")
        iteration_dir = eval_root / f"iter-{current_iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # -- Step 1: Dev benchmark --
        skip_dev = _should_skip_step("dev_benchmark", last_completed_step)
        if skip_dev:
            print("  [resume] skipping dev benchmark (already completed)")
        elif (
            args.skip_initial_benchmark
            and current_iteration == 1
            and last_completed_step is None
        ):
            cached = eval_root / "latest_run.json"
            if cached.exists():
                print("  [skip] using cached baseline for iteration 1")
                skip_dev = True

        if not skip_dev:
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
                "--run-label",
                "run",
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
                print("  [warn] dev benchmark failed; continuing with partial results")
                _save("dev_benchmark", dev_benchmark_failed=True)
            else:
                _save("dev_benchmark")
        else:
            _save("dev_benchmark")

        if stop_state.requested:
            break

        # -- Step 2: Cursor (inner) agent --
        if _should_skip_step("cursor", last_completed_step):
            print("  [resume] skipping cursor step (already completed)")
        else:
            eval_summary_path = _latest_eval_for_iteration(
                workdir_root=run_root,
                iteration=current_iteration,
            )
            if eval_summary_path is None:
                cached = eval_root / "latest_run.json"
                if cached.exists():
                    eval_summary_path = cached
                else:
                    raise RuntimeError("No eval summary available for cursor prompt.")

            # R4: pre-cursor snapshot
            pre_cursor_dir = (
                snapshot_root / f"iter-{current_iteration:04d}" / "pre-cursor"
            )
            if not pre_cursor_dir.exists():
                shutil.copytree(
                    src=workdir_root,
                    dst=pre_cursor_dir,
                    ignore=COPY_IGNORES,
                    dirs_exist_ok=False,
                )

            prompt_text = _render_prompt(
                template_path=template_path,
                iteration=current_iteration,
                run_root=run_root,
                workdir_root=workdir_root,
                eval_root=eval_root,
                snapshot_root=snapshot_root,
                eval_summary_path=eval_summary_path,
                context_length=context_length,
            )
            prompt_file = iteration_dir / "cursor_prompt.txt"
            prompt_file.write_text(prompt_text, encoding="utf-8")

            cursor_command = [
                "agent",
                "--print",
                "--force",
                "--trust",
                "--sandbox",
                "disabled",
                "--workspace",
                str(run_root),
            ]
            if args.cursor_model:
                cursor_command.extend(["--model", args.cursor_model])
            cursor_command.append(prompt_text)
            cursor_step = _run_command(command=cursor_command, cwd=run_root)
            if _is_transient_cursor_error(completed=cursor_step):
                print("Transient Cursor failure detected; retrying cursor step once.")
                cursor_step = _run_command(command=cursor_command, cwd=run_root)
            _record_step_output(
                target_path=iteration_dir / "outer_cursor_step.json",
                completed=cursor_step,
            )
            if cursor_step.returncode != 0:
                print(cursor_step.stdout)
                print(cursor_step.stderr, file=sys.stderr)
                return cursor_step.returncode

            _save("cursor")

        if stop_state.requested:
            break

        # -- Step 3: Validation --
        if _should_skip_step("validation", last_completed_step):
            print("  [resume] skipping validation (already completed)")
        else:
            test_command = [
                "uv",
                "run",
                "python",
                "-m",
                "unittest",
                "agent_evolve.test_interface_contract",
            ]
            test_env = os.environ.copy()
            pythonpath_parts = [str(repo_root / "cli"), str(run_root)]
            existing_pythonpath = test_env.get("PYTHONPATH")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            test_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
            test_step = _run_command(command=test_command, cwd=run_root, env=test_env)
            _record_step_output(
                target_path=iteration_dir / "outer_validation_step.json",
                completed=test_step,
            )
            if test_step.returncode != 0:
                print(test_step.stdout)
                print(test_step.stderr, file=sys.stderr)
                return test_step.returncode

            _save("validation")

        if stop_state.requested:
            break

        # -- Step 4: Eval benchmark (held-out) --
        if _should_skip_step("eval", last_completed_step):
            print("  [resume] skipping eval benchmark (already completed)")
        else:
            agent_py = workdir_root / "agent.py"
            current_hash = _file_hash(agent_py) if agent_py.exists() else None
            if current_hash and current_hash == last_eval_agent_hash:
                print(
                    "  [skip] agent.py unchanged since last eval; carrying forward score"
                )
                _save("eval")
            else:
                print(
                    f"--- eval benchmark (held-out) iteration {current_iteration} ---"
                )
                eval_benchmark_command = [
                    "uv",
                    "run",
                    "python",
                    "agent_evolve/run_recorded_benchmark.py",
                    "--iteration",
                    str(current_iteration),
                    "--runner",
                    str(args.eval_runner),
                    "--agent-key",
                    args.agent_key,
                    "--run-label",
                    "eval",
                ]
                if args.model_key:
                    eval_benchmark_command.extend(["--model-key", args.model_key])
                eval_benchmark_step = _run_command(
                    command=eval_benchmark_command, cwd=run_root
                )
                _record_step_output(
                    target_path=iteration_dir / "outer_eval_benchmark_step.json",
                    completed=eval_benchmark_step,
                )
                if eval_benchmark_step.returncode != 0:
                    print(eval_benchmark_step.stdout)
                    print(eval_benchmark_step.stderr, file=sys.stderr)
                    print("  [warn] eval benchmark failed; recording null score")
                    eval_score = None
                    _save("eval", eval_benchmark_failed=True)
                else:
                    eval_summary_for_state = _latest_eval_for_iteration(
                        workdir_root=run_root,
                        iteration=current_iteration,
                        label="eval",
                    )
                    if eval_summary_for_state and eval_summary_for_state.exists():
                        try:
                            eval_data = json.loads(
                                eval_summary_for_state.read_text(encoding="utf-8")
                            )
                            candidate = eval_data.get("reward_mean")
                            if isinstance(candidate, (int, float)):
                                eval_score = float(candidate)
                        except (json.JSONDecodeError, OSError):
                            pass

                    last_eval_agent_hash = current_hash
                    _save("eval")

        last_completed_step = None
        current_iteration += 1

    final_state = _build_state(
        stop_state=stop_state,
        args=args,
        run_root=run_root,
        current_iteration=current_iteration,
        last_completed_step=None,
        eval_score=eval_score,
        last_eval_agent_hash=last_eval_agent_hash,
        extra={"stopped_gracefully": stop_state.requested},
    )
    _save_state(state_path=state_path, payload=final_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
