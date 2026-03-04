from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from benchmark.runtime_config import build_runtime_cfg
import cli as cli_module

try:
    from terminal_bench.agents.agent_factory import (  # pyright: ignore[reportMissingImports]
        AgentFactory,
    )
    from terminal_bench.agents.base_agent import (  # pyright: ignore[reportMissingImports]
        AgentResult,
        BaseAgent,
    )
    from terminal_bench.agents.failure_mode import (  # pyright: ignore[reportMissingImports]
        FailureMode,
    )
    from terminal_bench.terminal.tmux_session import (  # pyright: ignore[reportMissingImports]
        TmuxSession,
    )
except Exception:
    # Keep this module importable in local test environments where
    # terminal-bench is not installed.
    AgentFactory = None

    class BaseAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def perform_task(
            self,
            instruction: str,
            session: "TmuxSession",
            logging_dir: Path | None = None,
        ) -> "AgentResult":
            del instruction
            del session
            del logging_dir
            raise NotImplementedError

    @dataclass
    class AgentResult:
        total_input_tokens: int = 0
        total_output_tokens: int = 0
        failure_mode: Any = "none"
        timestamped_markers: list[tuple[float, str]] | None = None

    class FailureMode:
        NONE = "none"
        UNKNOWN_AGENT_ERROR = "unknown_agent_error"

    class TmuxSession:
        pass


@dataclass
class HarborResolvedConfig:
    config_path: Path
    agent_key: str
    model_key: str
    loaded: Any


def resolve_harbor_config(
    *,
    config_path: Path | str | None = None,
    agent_key: str | None = None,
    model_key: str | None = None,
) -> HarborResolvedConfig:
    selected_path = Path(config_path) if config_path else cli_module.CONFIG_PATH
    selected_path = selected_path.expanduser()
    if not selected_path.is_absolute():
        selected_path = (Path.cwd() / selected_path).resolve()

    loaded = cli_module.load_config(selected_path)
    return HarborResolvedConfig(
        config_path=selected_path,
        agent_key=agent_key or loaded.default_agent,
        model_key=model_key or loaded.default_model,
        loaded=loaded,
    )


def _set_runtime_api_key_env(*, resolved: HarborResolvedConfig) -> None:
    model_cfg = resolved.loaded.models[resolved.model_key]
    env_name = cli_module._env_var_name(config_api_key=model_cfg.api_key)
    if not env_name:
        return

    api_key = cli_module.resolve_api_key(config_api_key=model_cfg.api_key)
    if not api_key:
        return

    os.environ[env_name] = api_key


def build_terminal_bench_agent_from_config(
    *,
    config_path: Path | str | None = None,
    agent_key: str | None = None,
    model_key: str | None = None,
) -> BaseAgent | None:
    resolved = resolve_harbor_config(
        config_path=config_path,
        agent_key=agent_key,
        model_key=model_key,
    )
    if AgentFactory is None:
        return None

    _set_runtime_api_key_env(resolved=resolved)
    runtime_cfg = build_runtime_cfg(
        cfg=resolved.loaded,
        agent_key=resolved.agent_key,
        model_key=resolved.model_key,
    )
    agent_kwargs = dict(runtime_cfg.agent_config)
    agent_kwargs.update(
        {
            "model_name": runtime_cfg.model.model,
            "api_base": runtime_cfg.model.api_base,
            "temperature": runtime_cfg.model.temperature
            if runtime_cfg.model.temperature is not None
            else 0.7,
        }
    )
    agent_class = AgentFactory.AGENT_NAME_TO_CLASS.get(resolved.agent_key)
    if agent_class is None:
        available = ", ".join(sorted(AgentFactory.AGENT_NAME_TO_CLASS.keys()))
        raise ValueError(
            f"Unsupported terminal-bench agent '{resolved.agent_key}'. "
            f"Available agent names: {available}"
        )

    return agent_class(**agent_kwargs)


class HarborTB2DefaultAgent(BaseAgent):  # pyright: ignore[reportGeneralTypeIssues]
    """
    Terminal-Bench import-path agent.

    This bridge resolves defaults from cli/config.json and delegates execution
    to an official terminal-bench agent instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            super().__init__(**kwargs)
        except TypeError:
            # Fallback shim base class may not accept kwargs.
            super().__init__()

        self._resolved = resolve_harbor_config(
            config_path=kwargs.get("config_path"),
            agent_key=kwargs.get("agent_key"),
            model_key=kwargs.get("model_key"),
        )
        self._tb_agent = build_terminal_bench_agent_from_config(
            config_path=self._resolved.config_path,
            agent_key=self._resolved.agent_key,
            model_key=self._resolved.model_key,
        )

    @staticmethod
    def name() -> str:
        return "small-agent-default"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        if self._tb_agent is None:
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                timestamped_markers=[],
            )

        try:
            return self._tb_agent.perform_task(
                instruction=instruction,
                session=session,
                logging_dir=logging_dir,
            )
        except Exception:
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                timestamped_markers=[],
            )
