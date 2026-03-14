# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path

try:
    from agent_evolve_v2.state import BenchmarkSummary, CriticItem, CriticSummary
except ModuleNotFoundError:
    from state_types import BenchmarkSummary, CriticItem, CriticSummary


def summarize_job(
    *,
    harbor_job_dir: Path,
    max_failure_items: int,
    max_success_items: int,
) -> tuple[BenchmarkSummary, CriticSummary]:
    aggregate_result_path = harbor_job_dir / "result.json"
    aggregate = _load_json(path=aggregate_result_path)
    stats = _as_dict(value=aggregate.get("stats"))
    evals = _as_dict(value=stats.get("evals"))
    eval_payload = next(iter(evals.values()), {})
    eval_stats = _as_dict(value=eval_payload)
    reward_stats = _as_dict(value=eval_stats.get("reward_stats"))
    reward_buckets = _as_dict(value=reward_stats.get("reward"))
    passed_trials = _string_list(value=reward_buckets.get("1.0"))
    failed_trials = _string_list(value=reward_buckets.get("0.0"))
    exception_types = {
        key: _string_list(value=value)
        for key, value in _as_dict(value=eval_stats.get("exception_stats")).items()
    }

    reward_mean = _extract_reward_mean(eval_stats=eval_stats)
    n_trials = _coerce_int(
        value=eval_stats.get("n_trials", stats.get("n_trials", 0)),
    )
    error_count = _coerce_int(
        value=eval_stats.get("n_errors", stats.get("n_errors", 0)),
    )

    benchmark_summary = BenchmarkSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        reward_mean=reward_mean,
        n_trials=n_trials,
        pass_count=len(passed_trials),
        failure_count=len(failed_trials),
        error_count=error_count,
        passed_trials=passed_trials,
        failed_trials=failed_trials,
        exception_types=exception_types,
    )

    critic_items = []
    sampled_failures = failed_trials[:max_failure_items]
    for priority, trial_name in enumerate(sampled_failures, start=1):
        critic_items.append(
            _build_failure_item(
                harbor_job_dir=harbor_job_dir,
                trial_name=trial_name,
                priority=priority,
            )
        )

    sampled_successes = passed_trials[:max_success_items]
    critic_summary = CriticSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        source_job_dir=str(harbor_job_dir),
        summary_markdown=_build_summary_markdown(
            benchmark_summary=benchmark_summary,
            critic_items=critic_items,
            sampled_successes=sampled_successes,
        ),
        items=critic_items,
    )
    return benchmark_summary, critic_summary


def write_critic_outputs(
    *,
    output_dir: Path,
    benchmark_summary: BenchmarkSummary,
    critic_summary: CriticSummary,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    critic_path = output_dir / "critic_summary.json"
    summary_path.write_text(
        json.dumps(asdict(benchmark_summary), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    critic_path.write_text(
        json.dumps(asdict(critic_summary), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "critic_summary.md").write_text(
        critic_summary.summary_markdown + "\n",
        encoding="utf-8",
    )
    return summary_path, critic_path


def _build_failure_item(
    *,
    harbor_job_dir: Path,
    trial_name: str,
    priority: int,
) -> CriticItem:
    trial_dir = harbor_job_dir / trial_name
    trial_result = _load_json(path=trial_dir / "result.json")
    task_name = str(trial_result.get("task_name", trial_name))
    verifier_output = _read_text(
        path=trial_dir / "verifier" / "test-stdout.txt",
        max_chars=2_000,
    )
    exception_text = _read_text(
        path=trial_dir / "exception.txt",
        max_chars=1_000,
    )
    stderr_text = _extract_agent_stderr(trial_result=trial_result)
    trajectory_snippet = _read_text(
        path=trial_dir / "agent" / "trajectory.json",
        max_chars=1_500,
    )

    failure_type = _classify_failure(
        verifier_output=verifier_output,
        exception_text=exception_text,
        stderr_text=stderr_text,
    )
    evidence_parts = []
    if verifier_output:
        evidence_parts.append("Verifier output:\n" + verifier_output)
    if exception_text:
        evidence_parts.append("Exception:\n" + exception_text)
    if stderr_text:
        evidence_parts.append("Agent stderr/stdout excerpt:\n" + stderr_text)
    if trajectory_snippet:
        evidence_parts.append("Trajectory excerpt:\n" + trajectory_snippet)

    return CriticItem(
        trial_name=trial_name,
        task_name=task_name,
        failure_type=failure_type,
        evidence="\n\n".join(evidence_parts).strip(),
        suspected_root_cause=_guess_root_cause(
            failure_type=failure_type,
            verifier_output=verifier_output,
            exception_text=exception_text,
            stderr_text=stderr_text,
        ),
        suggested_fix_direction=_suggest_fix_direction(
            failure_type=failure_type,
            verifier_output=verifier_output,
        ),
        priority=priority,
    )


def _build_summary_markdown(
    *,
    benchmark_summary: BenchmarkSummary,
    critic_items: list[CriticItem],
    sampled_successes: list[str],
) -> str:
    lines = [
        "# Critic Summary",
        "",
        f"- Reward mean: {benchmark_summary.reward_mean}",
        f"- Trials: {benchmark_summary.n_trials}",
        f"- Passed: {benchmark_summary.pass_count}",
        f"- Failed: {benchmark_summary.failure_count}",
        f"- Errors: {benchmark_summary.error_count}",
    ]
    if sampled_successes:
        lines.append(f"- Example successful trials: {', '.join(sampled_successes)}")

    lines.append("")
    lines.append("## Priority failures")
    if not critic_items:
        lines.append("")
        lines.append("No failed trials were sampled.")
        return "\n".join(lines)

    for item in critic_items:
        lines.extend(
            [
                "",
                f"### P{item.priority} - {item.task_name}",
                f"- Trial: `{item.trial_name}`",
                f"- Failure type: `{item.failure_type}`",
                f"- Suspected root cause: {item.suspected_root_cause}",
                f"- Suggested fix direction: {item.suggested_fix_direction}",
                "",
                "```text",
                item.evidence or "(no evidence captured)",
                "```",
            ]
        )

    return "\n".join(lines)


def _classify_failure(
    *,
    verifier_output: str,
    exception_text: str,
    stderr_text: str,
) -> str:
    combined = "\n".join((verifier_output, exception_text, stderr_text)).lower()
    if "timeout" in combined or "runtime budget exhaustion" in combined:
        return "timeout"
    if "traceback" in combined or "exception" in combined:
        return "agent_exception"
    if "assert" in combined or "expected" in combined or "failed" in combined:
        return "verifier_failure"
    if "syntax" in combined or "parse" in combined:
        return "configuration_error"
    return "benchmark_failure"


def _guess_root_cause(
    *,
    failure_type: str,
    verifier_output: str,
    exception_text: str,
    stderr_text: str,
) -> str:
    del exception_text
    lowered = "\n".join((verifier_output, stderr_text)).lower()
    if failure_type == "timeout":
        return "The agent spent too many turns or repeated failing actions without converging."
    if "permission denied" in lowered:
        return (
            "The harness attempted an invalid system action for the task environment."
        )
    if "no such file" in lowered or "not found" in lowered:
        return "The agent relied on an incorrect path or skipped an exploration step."
    if failure_type == "agent_exception":
        return "The harness produced an internal runtime failure or malformed command sequence."
    if failure_type == "configuration_error":
        return "The agent edited or generated config text in a way that broke syntax or formatting."
    return (
        "The harness strategy did not satisfy the verifier requirements for this task."
    )


def _suggest_fix_direction(
    *,
    failure_type: str,
    verifier_output: str,
) -> str:
    lowered = verifier_output.lower()
    if failure_type == "timeout":
        return "Bias the harness toward shorter plans, earlier verification, and faster pivots after repeated failures."
    if "log" in lowered or "config" in lowered:
        return "Improve file-edit verification and re-read configs after writing before restarting services."
    if "server" in lowered or "localhost" in lowered:
        return (
            "Add a stronger service-start verification step before declaring success."
        )
    if failure_type == "agent_exception":
        return "Harden the harness around command/result parsing and recovery after failed shell actions."
    return "Make the harness perform explicit verification against task requirements before finishing."


def _extract_reward_mean(*, eval_stats: dict[str, object]) -> float | None:
    metrics = eval_stats.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return None
    first_metric = metrics[0]
    if not isinstance(first_metric, dict):
        return None
    mean = first_metric.get("mean")
    if isinstance(mean, (int, float)):
        return float(mean)
    return None


def _extract_agent_stderr(*, trial_result: dict[str, object]) -> str:
    agent_result = _as_dict(value=trial_result.get("agent_result"))
    metadata = _as_dict(value=agent_result.get("metadata"))
    small_agent_result = _as_dict(value=metadata.get("small_agent_result"))
    stdout = str(small_agent_result.get("stdout", "") or "")
    stderr = str(small_agent_result.get("stderr", "") or "")
    combined = "\n".join(part for part in (stdout, stderr) if part).strip()
    return _truncate_text(text=combined, max_chars=2_000)


def _read_text(*, path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    return _truncate_text(text=path.read_text(encoding="utf-8"), max_chars=max_chars)


def _truncate_text(*, text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    half = max_chars // 2
    return stripped[:half] + "\n...[truncated]...\n" + stripped[-half:]


def _load_json(*, path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_dict(*, value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string_list(*, value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_int(*, value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0
