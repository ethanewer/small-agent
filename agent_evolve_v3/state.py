# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


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
    model_key: str
    aggregate_result_path: str
    harbor_job_dir: str
    benchmark_summary_path: str
    benchmark_stdout_path: str
    benchmark_stderr_path: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class PlanningOutput:
    selected_state_index: int
    plan: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, *, path: Path) -> "PlanningOutput":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Planner output must be a JSON object.")
        return cls.from_dict(data=payload)

    @classmethod
    def from_dict(cls, *, data: dict[str, Any]) -> "PlanningOutput":
        plan = _optional_text(value=data.get("plan"))
        if plan is None:
            raise ValueError("Planner output must include a non-empty 'plan'.")
        return cls(
            selected_state_index=int(data["selected_state_index"]),
            plan=plan,
        )


@dataclass
class AgentState:
    prev_path: str | None
    path: str
    refiner_workspace_path: str
    iteration: int
    baseline: str
    plan: str | None = None
    notes: str | None = None
    result: BenchmarkSummary | None = None
    official_benchmark: OfficialBenchmarkRun | None = None
    planner_selected_state_index: int | None = None
    planner_selected_iteration: int | None = None
    planner_prompt_artifact_path: str | None = None
    planner_output_artifact_path: str | None = None
    planner_summary: str | None = None
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, *, path: Path) -> "AgentState":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"State file must contain an object: {path}")
        return cls.from_dict(data=data)

    @classmethod
    def from_dict(cls, *, data: dict[str, Any]) -> "AgentState":
        raw_result = data.get("result")
        raw_official_benchmark = data.get("official_benchmark")
        result = (
            BenchmarkSummary(**raw_result) if isinstance(raw_result, dict) else None
        )
        official_benchmark = None
        if isinstance(raw_official_benchmark, dict):
            official_benchmark = OfficialBenchmarkRun(**raw_official_benchmark)

        return cls(
            prev_path=_optional_text(value=data.get("prev_path")),
            path=str(data["path"]),
            refiner_workspace_path=str(
                data.get("refiner_workspace_path", data.get("workspace_path"))
            ),
            iteration=int(data["iteration"]),
            baseline=str(data["baseline"]),
            plan=_optional_text(value=data.get("plan")),
            notes=_optional_text(value=data.get("notes")),
            result=result,
            official_benchmark=official_benchmark,
            planner_selected_state_index=_optional_int(
                value=data.get("planner_selected_state_index")
            ),
            planner_selected_iteration=_optional_int(
                value=data.get("planner_selected_iteration")
            ),
            planner_prompt_artifact_path=_optional_text(
                value=data.get("planner_prompt_artifact_path")
            ),
            planner_output_artifact_path=_optional_text(
                value=data.get("planner_output_artifact_path")
            ),
            planner_summary=_optional_text(value=data.get("planner_summary")),
            created_at_utc=str(
                data.get("created_at_utc", datetime.now(UTC).isoformat())
            ),
        )


def agent_state_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": [
            "path",
            "refiner_workspace_path",
            "iteration",
            "baseline",
        ],
        "properties": {
            "prev_path": {"type": ["string", "null"]},
            "path": {"type": "string"},
            "refiner_workspace_path": {"type": "string"},
            "iteration": {"type": "integer"},
            "baseline": {"type": "string"},
            "plan": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
            "result": {"type": ["object", "null"]},
            "official_benchmark": {"type": ["object", "null"]},
            "planner_selected_state_index": {"type": ["integer", "null"]},
            "planner_selected_iteration": {"type": ["integer", "null"]},
            "planner_prompt_artifact_path": {"type": ["string", "null"]},
            "planner_output_artifact_path": {"type": ["string", "null"]},
            "planner_summary": {"type": ["string", "null"]},
            "created_at_utc": {"type": "string"},
        },
    }


def planning_output_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["selected_state_index", "plan"],
        "properties": {
            "selected_state_index": {"type": "integer", "minimum": 0},
            "plan": {"type": "string", "minLength": 1},
        },
    }


def _optional_text(*, value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(*, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return int(text)
    return None
