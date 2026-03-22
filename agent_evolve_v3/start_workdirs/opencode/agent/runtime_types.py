# pyright: reportExplicitAny=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkspaceModelConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float | None = None
    context_length: int | None = None
    extra_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkspaceRuntimeConfig:
    model: WorkspaceModelConfig
    agent_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceRunResult:
    exit_code: int
    success: bool
    task_id: str = "adhoc"
