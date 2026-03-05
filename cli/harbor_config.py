from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from agents.interface import AgentModelConfig, AgentRuntimeConfig

CONFIG_PATH = Path(__file__).with_name("config.json")
ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass
class ConfigModelEntry:
    model: str
    api_base: str
    api_key: str | None
    temperature: float | None


@dataclass
class LoadedConfig:
    default_model: str
    models: dict[str, ConfigModelEntry]
    default_agent: str
    agents: dict[str, dict[str, Any]]
    verbosity: int
    max_turns: int
    max_wait_seconds: float


def _env_var_name(config_api_key: str | None) -> str | None:
    if not config_api_key:
        return None

    raw = config_api_key.strip()
    if not raw:
        return None

    if raw.startswith("$"):
        candidate = raw[1:]
        if ENV_NAME_PATTERN.fullmatch(candidate):
            return candidate

        return None

    if ENV_NAME_PATTERN.fullmatch(raw):
        return raw

    return None


def _shell_env_lookup(env_name: str) -> str | None:
    if not ENV_NAME_PATTERN.fullmatch(env_name):
        return None

    shell_command = f'source ~/.zshrc >/dev/null 2>&1; printf %s "${{{env_name}}}"'
    try:
        result = subprocess.run(
            ["zsh", "-ic", shell_command],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    candidate = result.stdout.strip()
    return candidate or None


def resolve_api_key(
    config_api_key: str | None,
    *,
    allow_shell_lookup: bool = True,
) -> str | None:
    env_name = _env_var_name(config_api_key=config_api_key)
    if env_name:
        direct_value = os.getenv(env_name)
        if direct_value:
            return direct_value

        if allow_shell_lookup:
            return _shell_env_lookup(env_name=env_name)

        return None

    if config_api_key and config_api_key.strip():
        return config_api_key.strip()

    return None


def _normalize_verbosity(value: int) -> int:
    if value == 3:
        return 1

    if value not in {0, 1}:
        raise ValueError("Verbosity must be one of: 0, 1.")

    return value


def load_config(path: Path) -> LoadedConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    data = json.loads(path.read_text())
    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("config.json must define a non-empty 'models' object.")

    models: dict[str, ConfigModelEntry] = {}
    for model_key, model_data in raw_models.items():
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError("Each key in 'models' must be a non-empty string.")

        if not isinstance(model_data, dict):
            raise ValueError(f"Model '{model_key}' must be an object.")

        model_name = str(model_data.get("model", "")).strip()
        api_base = str(model_data.get("api_base", "")).strip()
        if not model_name or not api_base:
            raise ValueError(
                f"Model '{model_key}' must include non-empty 'model' and 'api_base'."
            )

        raw_temperature = model_data.get("temperature")
        temperature = float(raw_temperature) if raw_temperature is not None else None
        models[model_key] = ConfigModelEntry(
            model=model_name,
            api_base=api_base,
            api_key=model_data.get("api_key"),
            temperature=temperature,
        )

    default_model = str(data.get("default_model", "")).strip()
    if not default_model:
        raise ValueError("config.json must define 'default_model'.")

    if default_model not in models:
        raise ValueError("default_model must match a key in models.")

    default_agent = str(data.get("default_agent", "terminus-2")).strip()
    raw_agents = data.get("agents")
    if raw_agents is None:
        raw_agents = {"terminus-2": {}}

    if not isinstance(raw_agents, dict) or not raw_agents:
        raise ValueError("config.json 'agents' must be a non-empty object.")

    agents: dict[str, dict[str, Any]] = {}
    for agent_key, agent_data in raw_agents.items():
        if not isinstance(agent_key, str) or not agent_key.strip():
            raise ValueError("Each key in 'agents' must be a non-empty string.")

        if agent_data is None:
            agent_data = {}

        if not isinstance(agent_data, dict):
            raise ValueError(f"Agent '{agent_key}' must be an object.")

        agents[agent_key] = dict(agent_data)

    if default_agent not in agents:
        raise ValueError("default_agent must match a key in agents.")

    return LoadedConfig(
        default_model=default_model,
        models=models,
        default_agent=default_agent,
        agents=agents,
        verbosity=_normalize_verbosity(int(data.get("verbosity", 1))),
        max_turns=int(data.get("max_turns", 50)),
        max_wait_seconds=float(data.get("max_wait_seconds", 60.0)),
    )


def build_runtime_config(
    *,
    config: LoadedConfig,
    agent_key: str,
    model_key: str,
    allow_shell_lookup: bool = True,
) -> AgentRuntimeConfig:
    selected = config.models[model_key]
    resolved_api_key = resolve_api_key(
        config_api_key=selected.api_key,
        allow_shell_lookup=allow_shell_lookup,
    )
    if not resolved_api_key:
        raise ValueError("Missing API key for selected model.")

    agent_options = {
        "verbosity": config.verbosity,
        "max_turns": config.max_turns,
        "max_wait_seconds": config.max_wait_seconds,
        **dict(config.agents.get(agent_key, {})),
    }
    return AgentRuntimeConfig(
        agent_key=agent_key,
        model=AgentModelConfig(
            model=selected.model,
            api_base=selected.api_base,
            api_key=resolved_api_key,
            temperature=selected.temperature,
        ),
        agent_config=agent_options,
    )
