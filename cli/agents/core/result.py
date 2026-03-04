from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    exit_code: int
    success: bool
    task_id: str = "adhoc"
    final_message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    trace_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
