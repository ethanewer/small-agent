# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from agent_evolve_v3.state import BenchmarkSummary


def summarize_benchmark_job(*, harbor_job_dir: Path) -> BenchmarkSummary:
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

    return BenchmarkSummary(
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
