from __future__ import annotations

from agents.interface import Agent
from agents.qwen import QwenHeadlessAgent
from agents.terminus2.agent import Terminus2Agent


def available_agents() -> dict[str, Agent]:
    return {
        "terminus-2": Terminus2Agent(),
        "qwen": QwenHeadlessAgent(),
    }


def get_agent(agent_key: str) -> Agent:
    agents = available_agents()
    try:
        return agents[agent_key]
    except KeyError as err:
        supported = ", ".join(sorted(agents.keys()))
        raise ValueError(
            f"Unknown agent '{agent_key}'. Available agents: {supported}"
        ) from err
