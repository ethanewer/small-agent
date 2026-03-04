from __future__ import annotations

from typing import Any

from agents.interface import AgentModelConfig, AgentRuntimeConfig
import cli as cli_module


def build_runtime_cfg(
    *, cfg: Any, agent_key: str, model_key: str
) -> AgentRuntimeConfig:
    model_cfg = cfg.models[model_key]
    resolved_api_key = cli_module.resolve_api_key(model_cfg.api_key)
    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for model '{model_key}'. Set env var or literal api_key."
        )

    agent_options = {
        "verbosity": cfg.verbosity,
        "max_turns": cfg.max_turns,
        "max_wait_seconds": cfg.max_wait_seconds,
        **dict(cfg.agents.get(agent_key, {})),
    }
    return AgentRuntimeConfig(
        agent_key=agent_key,
        model=AgentModelConfig(
            model=model_cfg.model,
            api_base=model_cfg.api_base,
            api_key=resolved_api_key,
            temperature=model_cfg.temperature,
        ),
        agent_config=agent_options,
    )
