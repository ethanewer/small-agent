from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskContext:
    dataset: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    instruction: str
    task_id: str = "adhoc"
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    context: TaskContext = field(default_factory=TaskContext)

    @classmethod
    def from_instruction(
        cls,
        *,
        instruction: str,
        task_id: str = "adhoc",
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        return cls(
            instruction=instruction,
            task_id=task_id,
            metadata=metadata or {},
        )
