from __future__ import annotations

from agents.agent_types import AgentRunner

AVAILABLE_AGENTS = ["terminus2"]


def get_agent(name: str) -> AgentRunner:
    if name == "terminus2":
        from agents.terminus2 import run

        return run

    available = ", ".join(AVAILABLE_AGENTS)
    raise KeyError(f"Unknown agent {name!r}. Available: {available}")
