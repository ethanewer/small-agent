from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
        default=Path("harbor/run_dev_benchmark.sh"),
    )
    parser.add_argument("--agent-key", type=str, default="terminus-2")
    parser.add_argument("--model-key", type=str, default=None)
    parser.add_argument("--cursor-model", type=str, default=None)
    parser.add_argument(
        "--eval-runner",
        type=Path,
        default=Path("harbor/run_small_benchmark.sh"),
        help="Runner script for the eval benchmark (run between outer loop iterations).",
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
    parser.add_argument(
        "--runner-args",
        type=str,
        default="",
        help="Extra arguments forwarded to the dev runner script.",
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


def _resolve_cursor_model(*, config_path: Path) -> str | None:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        value = data.get("default_cursor_model")
        if isinstance(value, str) and value:
            return value
    except (json.JSONDecodeError, OSError):
        pass

    return None


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


def _find_eval_harbor_result(*, eval_root: Path, iteration: int) -> Path | None:
    """Locate the aggregate result.json for an iteration's eval benchmark."""
    eval_dir = eval_root / f"iter-{iteration:04d}" / "eval-0001" / "harbor_job"
    if not eval_dir.exists():
        return None

    for job_dir in sorted(eval_dir.iterdir()):
        candidate = job_dir / "result.json"
        if candidate.is_file():
            return candidate

    return None


def _parse_harbor_result(result_path: Path) -> tuple[float | None, int | None]:
    """Return (reward_mean, n_trials) from a harbor aggregate result.json."""
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        stats = data.get("stats", {})
        n_trials = stats.get("n_trials")
        evals = stats.get("evals", {})
        for eval_data in evals.values():
            metrics = eval_data.get("metrics", [])
            if metrics and isinstance(metrics[0], dict):
                mean = metrics[0].get("mean")
                if isinstance(mean, (int, float)):
                    return float(mean), n_trials
    except (json.JSONDecodeError, OSError):
        pass

    return None, None


def _best_dev_score(*, eval_root: Path, iteration: int) -> float | None:
    """Return the best dev run reward_mean across all runs for an iteration."""
    iter_dir = eval_root / f"iter-{iteration:04d}"
    if not iter_dir.exists():
        return None

    best: float | None = None
    for run_dir in sorted(iter_dir.glob("run-*")):
        harbor_dir = run_dir / "harbor_job"
        if not harbor_dir.exists():
            continue
        for job_dir in sorted(harbor_dir.iterdir()):
            result_path = job_dir / "result.json"
            if result_path.is_file():
                score, _ = _parse_harbor_result(result_path)
                if score is not None and (best is None or score > best):
                    best = score

    return best


def _agent_line_count(*, snapshot_root: Path, iteration: int) -> int | None:
    """Count lines in the eval snapshot's agent.py."""
    agent_path = snapshot_root / f"iter-{iteration:04d}" / "eval-0001" / "agent.py"
    if not agent_path.exists():
        agent_path = snapshot_root / f"iter-{iteration:04d}" / "pre-cursor" / "agent.py"
    if not agent_path.exists():
        return None

    try:
        return len(agent_path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return None


_ARCH_HEADING_RE = re.compile(r"^###?\s+Architecture\s+[\d.]+[:\s]+(.+)", re.MULTILINE)


def _extract_architecture_label(*, snapshot_root: Path, iteration: int) -> str:
    """Pull the latest architecture heading from a snapshot's NOTES.md."""
    for label in ("eval-0001", "pre-cursor"):
        notes_path = snapshot_root / f"iter-{iteration:04d}" / label / "NOTES.md"
        if notes_path.exists():
            try:
                text = notes_path.read_text(encoding="utf-8")
            except OSError:
                continue
            matches = _ARCH_HEADING_RE.findall(text)
            if matches:
                raw = matches[-1].strip()
                raw = re.sub(r"\s*\(current\)", "", raw, flags=re.IGNORECASE)
                return raw

    return ""


def _extract_notes_summary(
    *, snapshot_root: Path, iteration: int, max_lines: int = 10
) -> str:
    """Return a brief summary from a snapshot's NOTES.md."""
    for label in ("eval-0001", "pre-cursor"):
        notes_path = snapshot_root / f"iter-{iteration:04d}" / label / "NOTES.md"
        if notes_path.exists():
            try:
                text = notes_path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = text.splitlines()
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + "\n..."
            return text

    return "No notes available."


@dataclass
class IterationRecord:
    iteration: int
    eval_score: float | None
    dev_best: float | None
    n_trials: int | None
    line_count: int | None
    architecture: str
    snapshot_agent_path: str | None


def collect_iteration_records(
    *,
    run_root: Path,
    up_to_iteration: int,
) -> list[IterationRecord]:
    """Gather summary data for iterations 1..up_to_iteration."""
    eval_root = run_root / "eval"
    snapshot_root = run_root / "snapshots"
    records: list[IterationRecord] = []

    for it in range(1, up_to_iteration + 1):
        result_path = _find_eval_harbor_result(eval_root=eval_root, iteration=it)
        eval_score: float | None = None
        n_trials: int | None = None
        if result_path is not None:
            eval_score, n_trials = _parse_harbor_result(result_path)

        dev_best = _best_dev_score(eval_root=eval_root, iteration=it)
        line_count = _agent_line_count(snapshot_root=snapshot_root, iteration=it)
        architecture = _extract_architecture_label(
            snapshot_root=snapshot_root, iteration=it
        )

        agent_path: str | None = None
        for label in ("eval-0001", "pre-cursor"):
            candidate = snapshot_root / f"iter-{it:04d}" / label / "agent.py"
            if candidate.exists():
                agent_path = str(candidate)
                break

        records.append(
            IterationRecord(
                iteration=it,
                eval_score=eval_score,
                dev_best=dev_best,
                n_trials=n_trials,
                line_count=line_count,
                architecture=architecture,
                snapshot_agent_path=agent_path,
            )
        )

    return records


def _baseline_record(*, run_root: Path) -> IterationRecord:
    """Build a synthetic record for the baseline (iteration 0)."""
    snapshot_root = run_root / "snapshots"
    agent_path = snapshot_root / "iter-0001" / "pre-cursor" / "agent.py"
    line_count: int | None = None
    if agent_path.exists():
        try:
            line_count = len(agent_path.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass

    return IterationRecord(
        iteration=0,
        eval_score=None,
        dev_best=None,
        n_trials=None,
        line_count=line_count,
        architecture="Baseline",
        snapshot_agent_path=str(agent_path) if agent_path.exists() else None,
    )


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _build_scoreboard_text(
    *,
    run_root: Path,
    up_to_iteration: int,
    baseline_eval_score: float | None = None,
) -> str:
    """Build the scoreboard markdown table for the prompt."""
    baseline = _baseline_record(run_root=run_root)
    if baseline_eval_score is not None:
        baseline.eval_score = baseline_eval_score

    records = collect_iteration_records(
        run_root=run_root, up_to_iteration=up_to_iteration
    )
    all_records = [baseline, *records]

    best_score: float | None = None
    best_iter: int | None = None
    for rec in all_records:
        if rec.eval_score is not None and (
            best_score is None or rec.eval_score > best_score
        ):
            best_score = rec.eval_score
            best_iter = rec.iteration

    lines: list[str] = []
    lines.append(
        "## Eval score history (held-out benchmark — you cannot see these tasks)\n"
    )
    if best_iter is not None and best_score is not None:
        passed = ""
        for rec in all_records:
            if rec.iteration == best_iter and rec.n_trials:
                n_passed = round(best_score * rec.n_trials)
                passed = f" ({n_passed}/{rec.n_trials} passed)"
                break
        lines.append(
            f"Best iteration so far: **Iter {best_iter}** with eval score "
            f"**{best_score:.2f}**{passed}\n"
        )

    lines.append("| Iter | Eval Score | Dev Best | Lines | Architecture |")
    lines.append("|------|-----------|----------|-------|--------------|")
    for rec in all_records:
        lines.append(
            f"| {rec.iteration} "
            f"| {_fmt_score(rec.eval_score)} "
            f"| {_fmt_score(rec.dev_best)} "
            f"| {rec.line_count or 'N/A'} "
            f"| {rec.architecture or 'N/A'} |"
        )

    latest_with_score = [r for r in records if r.eval_score is not None]
    if best_score is not None and len(latest_with_score) >= 2:
        consecutive_decline = 0
        for rec in reversed(latest_with_score):
            if rec.eval_score is not None and rec.eval_score < best_score:
                consecutive_decline += 1
            else:
                break

        if consecutive_decline >= 2:
            lines.append("")
            lines.append(
                f"WARNING: Eval score has declined for {consecutive_decline} "
                f"consecutive iterations. Your recent changes are hurting "
                f"held-out performance. Consider reverting to the architecture "
                f"from iteration {best_iter} and making smaller, more targeted "
                f"changes."
            )

    return "\n".join(lines)


def _build_snapshot_index(
    *,
    run_root: Path,
    up_to_iteration: int,
    baseline_eval_score: float | None = None,
) -> str:
    """Build the snapshot index for the prompt."""
    baseline = _baseline_record(run_root=run_root)
    if baseline_eval_score is not None:
        baseline.eval_score = baseline_eval_score

    records = collect_iteration_records(
        run_root=run_root, up_to_iteration=up_to_iteration
    )
    all_records = [baseline, *records]

    best_score: float | None = None
    best_iter: int | None = None
    for rec in all_records:
        if rec.eval_score is not None and (
            best_score is None or rec.eval_score > best_score
        ):
            best_score = rec.eval_score
            best_iter = rec.iteration

    lines: list[str] = []
    lines.append("## Prior agent snapshots (read-only, for reference)\n")
    lines.append("| Iter | Eval Score | Snapshot Path |")
    lines.append("|------|-----------|--------------|")
    for rec in all_records:
        if rec.snapshot_agent_path:
            lines.append(
                f"| {rec.iteration} "
                f"| {_fmt_score(rec.eval_score)} "
                f"| `{rec.snapshot_agent_path}` |"
            )

    if best_iter is not None:
        best_rec = next((r for r in all_records if r.iteration == best_iter), None)
        if best_rec and best_rec.snapshot_agent_path:
            lines.append("")
            lines.append(
                "To compare your current agent against the best-performing version:"
            )
            lines.append(f"  `diff {best_rec.snapshot_agent_path} agent.py`")

    return "\n".join(lines)


def update_scoreboard(
    *,
    run_root: Path,
    up_to_iteration: int,
    baseline_eval_score: float | None = None,
) -> None:
    """Regenerate SCOREBOARD.md and scoreboard.json from all completed iterations."""
    scoreboard_text = _build_scoreboard_text(
        run_root=run_root,
        up_to_iteration=up_to_iteration,
        baseline_eval_score=baseline_eval_score,
    )
    (run_root / "SCOREBOARD.md").write_text(scoreboard_text + "\n", encoding="utf-8")

    baseline = _baseline_record(run_root=run_root)
    if baseline_eval_score is not None:
        baseline.eval_score = baseline_eval_score

    records = collect_iteration_records(
        run_root=run_root, up_to_iteration=up_to_iteration
    )
    all_records = [baseline, *records]

    json_data = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "iterations": [
            {
                "iteration": rec.iteration,
                "eval_score": rec.eval_score,
                "dev_best": rec.dev_best,
                "n_trials": rec.n_trials,
                "line_count": rec.line_count,
                "architecture": rec.architecture,
                "snapshot_agent_path": rec.snapshot_agent_path,
            }
            for rec in all_records
        ],
    }
    (run_root / "scoreboard.json").write_text(
        json.dumps(json_data, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def seed_notes_with_history(
    *,
    workdir_root: Path,
    run_root: Path,
    up_to_iteration: int,
    baseline_eval_score: float | None = None,
) -> None:
    """Prepend structured iteration history to the workspace NOTES.md."""
    snapshot_root = run_root / "snapshots"
    scoreboard_text = _build_scoreboard_text(
        run_root=run_root,
        up_to_iteration=up_to_iteration,
        baseline_eval_score=baseline_eval_score,
    )

    history_parts: list[str] = []
    history_parts.append("<!-- BEGIN ITERATION HISTORY (auto-generated) -->\n")
    history_parts.append(scoreboard_text)
    history_parts.append("")

    records = collect_iteration_records(
        run_root=run_root, up_to_iteration=up_to_iteration
    )
    if records:
        history_parts.append("## Per-iteration summaries\n")
        for rec in records:
            header = f"### Iter {rec.iteration}"
            if rec.architecture:
                header += f" — {rec.architecture}"
            header += f" (eval: {_fmt_score(rec.eval_score)})"
            history_parts.append(header)
            summary = _extract_notes_summary(
                snapshot_root=snapshot_root, iteration=rec.iteration
            )
            history_parts.append(summary)
            history_parts.append("")

    history_parts.append("<!-- END ITERATION HISTORY -->\n")
    history_block = "\n".join(history_parts)

    notes_path = workdir_root / "NOTES.md"
    existing = ""
    if notes_path.exists():
        try:
            existing = notes_path.read_text(encoding="utf-8")
        except OSError:
            pass

    existing = re.sub(
        r"<!-- BEGIN ITERATION HISTORY \(auto-generated\) -->.*?"
        r"<!-- END ITERATION HISTORY -->\n?",
        "",
        existing,
        flags=re.DOTALL,
    )
    existing = existing.lstrip("\n")

    notes_path.write_text(history_block + "\n" + existing, encoding="utf-8")


def _render_prompt(
    *,
    template_path: Path,
    iteration: int,
    workdir_root: Path,
    eval_root: Path,
    snapshot_root: Path,
    eval_summary_path: Path,
    context_length: int,
    run_root: Path,
    baseline_eval_score: float | None = None,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    eval_artifacts_path = eval_summary_path.parent
    dev_score, dev_trials = _read_eval_summary_fields(eval_summary_path)

    prior_iteration = iteration - 1
    scoreboard = _build_scoreboard_text(
        run_root=run_root,
        up_to_iteration=max(prior_iteration, 0),
        baseline_eval_score=baseline_eval_score,
    )
    snapshot_index = _build_snapshot_index(
        run_root=run_root,
        up_to_iteration=max(prior_iteration, 0),
        baseline_eval_score=baseline_eval_score,
    )

    return text.format(
        iteration=iteration,
        iteration_padded=f"{iteration:04d}",
        workdir_root=workdir_root,
        eval_root=eval_root,
        snapshot_root=snapshot_root,
        eval_summary_path=eval_summary_path,
        eval_artifacts_path=eval_artifacts_path,
        context_length=context_length,
        dev_score=dev_score,
        dev_trials=dev_trials,
        scoreboard=scoreboard,
        snapshot_index=snapshot_index,
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
    baseline_eval_score: float | None = None,
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
        "baseline_eval_score": baseline_eval_score,
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
    repo_root = root_dir.parent
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
    config_path = repo_root / "config.json"
    context_length = _resolve_context_length(
        config_path=config_path, model_key=args.model_key
    )
    if not args.cursor_model:
        args.cursor_model = _resolve_cursor_model(config_path=config_path)

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

    baseline_eval_score: float | None = None
    raw_baseline = state.get("baseline_eval_score")
    if isinstance(raw_baseline, (int, float)):
        baseline_eval_score = float(raw_baseline)

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
            baseline_eval_score=baseline_eval_score,
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
            if args.runner_args:
                benchmark_command.extend(["--runner-args", args.runner_args])
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

            seed_notes_with_history(
                workdir_root=workdir_root,
                run_root=run_root,
                up_to_iteration=current_iteration - 1,
                baseline_eval_score=baseline_eval_score,
            )

            prompt_text = _render_prompt(
                template_path=template_path,
                iteration=current_iteration,
                workdir_root=workdir_root,
                eval_root=eval_root,
                snapshot_root=snapshot_root,
                eval_summary_path=eval_summary_path,
                context_length=context_length,
                run_root=run_root,
                baseline_eval_score=baseline_eval_score,
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
                str(workdir_root),
            ]
            if args.cursor_model:
                cursor_command.extend(["--model", args.cursor_model])
            cursor_command.append(prompt_text)
            cursor_step = _run_command(command=cursor_command, cwd=workdir_root)
            if _is_transient_cursor_error(completed=cursor_step):
                print("Transient Cursor failure detected; retrying cursor step once.")
                cursor_step = _run_command(command=cursor_command, cwd=workdir_root)
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
            pythonpath_parts = [str(repo_root), str(run_root)]
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
                    if current_iteration == 1 and eval_score is not None:
                        baseline_eval_score = eval_score

                    _save("eval")

            update_scoreboard(
                run_root=run_root,
                up_to_iteration=current_iteration,
                baseline_eval_score=baseline_eval_score,
            )

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
        baseline_eval_score=baseline_eval_score,
        extra={"stopped_gracefully": stop_state.requested},
    )
    _save_state(state_path=state_path, payload=final_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
