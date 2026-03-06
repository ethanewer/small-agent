from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "reasoning",
    "command",
    "command_output",
    "tool_call",
    "tool_result",
    "issue",
    "done",
    "stopped",
]


@dataclass
class AgentEvent:
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    turn: int | None = None
