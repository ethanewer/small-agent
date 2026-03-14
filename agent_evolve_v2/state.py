# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


@dataclass
class CriticItem:
    trial_name: str
    task_name: str
    failure_type: str
    evidence: str
    suspected_root_cause: str
    suggested_fix_direction: str
    priority: int


@dataclass
class CriticSummary:
    created_at_utc: str
    source_job_dir: str
    summary_markdown: str
    items: list[CriticItem] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    created_at_utc: str
    aggregate_result_path: str
    harbor_job_dir: str
    reward_mean: float | None
    n_trials: int
    pass_count: int
    failure_count: int
    error_count: int
    passed_trials: list[str] = field(default_factory=list)
    failed_trials: list[str] = field(default_factory=list)
    exception_types: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class OfficialBenchmarkRun:
    fingerprint: str
    model_key: str
    canonical_run_dir: str
    canonical_manifest_path: str
    aggregate_result_path: str
    harbor_job_dir: str
    benchmark_summary_path: str
    benchmark_stdout_path: str
    benchmark_stderr_path: str
    included_files: list[str] = field(default_factory=list)
    cache_hit: bool = False
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AgentState:
    prev_path: str | None
    path: str
    refiner_workspace_path: str
    iteration: int
    baseline: str
    critic_workspace_path: str | None = None
    plan: str | None = None
    notes: str | None = None
    result: BenchmarkSummary | None = None
    critic: CriticSummary | None = None
    official_benchmark: OfficialBenchmarkRun | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def workspace_path(self) -> str:
        return self.refiner_workspace_path

    def save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(
            self.to_json() + "\n",
            encoding="utf-8",
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, *, path: Path) -> AgentState:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data=data)

    @classmethod
    def from_dict(cls, *, data: dict[str, Any]) -> AgentState:
        raw_result = data.get("result")
        raw_critic = data.get("critic")
        raw_official_benchmark = data.get("official_benchmark")
        result = (
            BenchmarkSummary(**raw_result) if isinstance(raw_result, dict) else None
        )
        critic = None
        if isinstance(raw_critic, dict):
            raw_items = raw_critic.get("items", [])
            items = []
            if isinstance(raw_items, list):
                for item in raw_items:
                    if isinstance(item, dict):
                        items.append(CriticItem(**item))
            critic = CriticSummary(
                created_at_utc=str(raw_critic.get("created_at_utc", "")),
                source_job_dir=str(raw_critic.get("source_job_dir", "")),
                summary_markdown=str(raw_critic.get("summary_markdown", "")),
                items=items,
            )
        official_benchmark = None
        if isinstance(raw_official_benchmark, dict):
            official_benchmark = OfficialBenchmarkRun(**raw_official_benchmark)

        return cls(
            prev_path=data.get("prev_path"),
            path=str(data["path"]),
            refiner_workspace_path=str(
                data.get("refiner_workspace_path", data.get("workspace_path"))
            ),
            iteration=int(data["iteration"]),
            baseline=str(data["baseline"]),
            critic_workspace_path=_optional_text(
                value=data.get("critic_workspace_path")
            ),
            plan=_optional_text(value=data.get("plan")),
            notes=_optional_text(value=data.get("notes")),
            result=result,
            critic=critic,
            official_benchmark=official_benchmark,
            created_at_utc=str(
                data.get("created_at_utc", datetime.now(UTC).isoformat())
            ),
        )


def _optional_text(*, value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
