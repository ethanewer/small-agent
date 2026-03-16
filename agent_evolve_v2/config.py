# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_BASELINES = ("liteforge", "terminus2")


@dataclass(frozen=True)
class RunSpec:
    name: str
    baseline: str
    model_key: str
    cursor_model: str
    iterations: int = 25
    max_critic_failures: int = 5
    max_critic_successes: int = 1
    random_seed: int = 0
    benchmark_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunsConfig:
    default_run: str
    runs: dict[str, RunSpec]

    def get_run(self, *, run_name: str | None) -> RunSpec:
        selected_name = run_name or self.default_run
        try:
            return self.runs[selected_name]
        except KeyError as err:
            supported = ", ".join(sorted(self.runs))
            raise ValueError(
                f"Unknown run '{selected_name}'. Available runs: {supported}"
            ) from err


def load_runs_config(*, path: Path) -> RunsConfig:
    payload = _load_yamlish(path=path)
    config_name = path.name
    if not isinstance(payload, dict):
        raise ValueError(f"{config_name} must contain a top-level object.")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, dict) or not raw_runs:
        raise ValueError(f"{config_name} must define a non-empty 'runs' mapping.")

    default_run = str(payload.get("default_run", "")).strip()
    if not default_run:
        raise ValueError(f"{config_name} must define 'default_run'.")

    runs: dict[str, RunSpec] = {}
    for run_name, run_payload in raw_runs.items():
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("Each run name must be a non-empty string.")
        if not isinstance(run_payload, dict):
            raise ValueError(f"Run '{run_name}' must be an object.")

        baseline = str(run_payload.get("baseline", "")).strip()
        if baseline not in SUPPORTED_BASELINES:
            supported = ", ".join(SUPPORTED_BASELINES)
            raise ValueError(
                f"Run '{run_name}' uses unsupported baseline '{baseline}'. "
                f"Supported baselines: {supported}"
            )

        model_key = str(run_payload.get("model_key", "")).strip()
        cursor_model = str(run_payload.get("cursor_model", "")).strip()
        if not model_key:
            raise ValueError(f"Run '{run_name}' must define 'model_key'.")
        if not cursor_model:
            raise ValueError(f"Run '{run_name}' must define 'cursor_model'.")
        raw_benchmark_tasks = run_payload.get("benchmark_tasks", [])
        if raw_benchmark_tasks is None:
            benchmark_tasks: tuple[str, ...] = ()
        elif isinstance(raw_benchmark_tasks, list):
            benchmark_tasks = tuple(
                str(task_name).strip() for task_name in raw_benchmark_tasks
            )
            if any(not task_name for task_name in benchmark_tasks):
                raise ValueError(
                    f"Run '{run_name}' benchmark_tasks must contain only "
                    "non-empty strings."
                )
        else:
            raise ValueError(
                f"Run '{run_name}' benchmark_tasks must be a list of strings."
            )

        runs[run_name] = RunSpec(
            name=run_name,
            baseline=baseline,
            model_key=model_key,
            cursor_model=cursor_model,
            iterations=max(1, int(run_payload.get("iterations", 25))),
            max_critic_failures=max(
                1,
                int(run_payload.get("max_critic_failures", 5)),
            ),
            max_critic_successes=max(
                0,
                int(run_payload.get("max_critic_successes", 1)),
            ),
            random_seed=int(run_payload.get("random_seed", 0)),
            benchmark_tasks=benchmark_tasks,
        )

    if default_run not in runs:
        raise ValueError("default_run must match a key inside 'runs'.")

    return RunsConfig(default_run=default_run, runs=runs)


def _load_yamlish(*, path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        # When PyYAML isn't installed, JSON config files still parse cleanly.
        return json.loads(text)
