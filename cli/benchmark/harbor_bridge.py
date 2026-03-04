from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.core.task import Task
from agents.registry import get_agent
from benchmark.adapters import AgentBenchmarkAdapter
from benchmark.runtime_config import build_runtime_cfg
import cli as cli_module

try:
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
    class BaseAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

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
    loaded = cli_module.load_config(selected_path)
    return HarborResolvedConfig(
        config_path=selected_path,
        agent_key=agent_key or loaded.default_agent,
        model_key=model_key or loaded.default_model,
        loaded=loaded,
    )


def build_adapter_from_config(
    *,
    config_path: Path | str | None = None,
    agent_key: str | None = None,
    model_key: str | None = None,
    console: Any | None = None,
) -> AgentBenchmarkAdapter:
    resolved = resolve_harbor_config(
        config_path=config_path,
        agent_key=agent_key,
        model_key=model_key,
    )
    runtime_cfg = build_runtime_cfg(
        cfg=resolved.loaded,
        agent_key=resolved.agent_key,
        model_key=resolved.model_key,
    )
    return AgentBenchmarkAdapter(
        agent=get_agent(resolved.agent_key),
        runtime_cfg=runtime_cfg,
        console=console,
    )


class HarborTB2DefaultAgent(BaseAgent):  # pyright: ignore[reportGeneralTypeIssues]
    """
    Terminal-Bench import-path agent.

    This bridge resolves defaults from cli/config.json and delegates execution
    to the existing small-agent runtime adapter.
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
        self._adapter = build_adapter_from_config(
            config_path=self._resolved.config_path,
            agent_key=self._resolved.agent_key,
            model_key=self._resolved.model_key,
            console=None,
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
        del session
        del logging_dir
        run_result = self._adapter.run_sync(
            task=Task.from_instruction(
                instruction=instruction,
                task_id="harbor-import-path-task",
            )
        )

        input_tokens = int(
            run_result.metrics.get("input_tokens")
            or run_result.metrics.get("prompt_tokens")
            or 0
        )
        output_tokens = int(
            run_result.metrics.get("output_tokens")
            or run_result.metrics.get("completion_tokens")
            or 0
        )
        failure_mode = (
            FailureMode.NONE if run_result.success else FailureMode.UNKNOWN_AGENT_ERROR
        )
        return AgentResult(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            failure_mode=failure_mode,
            timestamped_markers=[],
        )
