from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rich.console import Console


@dataclass
class AgentModelConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float | None = None


@dataclass
class AgentRuntimeConfig:
    agent_key: str
    model: AgentModelConfig
    agent_config: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    def run(
        self,
        instruction: str,
        cfg: AgentRuntimeConfig,
        console: Console,
    ) -> int: ...
