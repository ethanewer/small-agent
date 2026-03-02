from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from agents.interface import AgentModelConfig, AgentRuntimeConfig
from agents.registry import available_agents, get_agent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

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


@dataclass
class InteractiveCommandResult:
    instruction: str
    selected_model: str | None = None
    selected_agent: str | None = None
    updated_verbosity: int | None = None
    handled: bool = False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-agent wrapper CLI")
    parser.add_argument(
        "instruction",
        nargs="*",
        help="Instruction for the agent. If omitted, interactive prompt is used.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 3],
        default=None,
        help="0: one line per tool call, 1: full tool inputs/responses, 3: + reasoning",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.json with model/API settings.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns from config.json for this run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key from config.models to run with.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent key from config.agents to run with.",
    )

    return parser.parse_args(argv)


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


def resolve_api_key(config_api_key: str | None) -> str | None:
    env_name = _env_var_name(config_api_key=config_api_key)
    if env_name:
        direct_value = os.getenv(env_name)
        if direct_value:
            return direct_value

        return _shell_env_lookup(env_name=env_name)

    if config_api_key and config_api_key.strip():
        return config_api_key.strip()

    return None


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
        verbosity=int(data.get("verbosity", 1)),
        max_turns=int(data.get("max_turns", 50)),
        max_wait_seconds=float(data.get("max_wait_seconds", 60.0)),
    )


def select_model_dialog(console: Console, config: LoadedConfig) -> str:
    model_keys = list(config.models.keys())
    numbered_lines = []
    for index, model_key in enumerate(model_keys, start=1):
        numbered_lines.append(f"{index}. {model_key}")

    console.print(
        Panel("\n".join(numbered_lines), title="Available Models", border_style="cyan")
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter model number[/bold]").strip()
        if not raw_choice:
            continue
        if not raw_choice.isdigit():
            console.print(Panel("Please enter a valid number.", border_style="yellow"))
            continue

        selected_index = int(raw_choice)
        if selected_index < 1 or selected_index > len(model_keys):
            console.print(Panel("Model number out of range.", border_style="yellow"))
            continue

        return model_keys[selected_index - 1]


def select_agent_dialog(console: Console, config: LoadedConfig) -> str:
    agent_keys = list(config.agents.keys())
    numbered_lines = []
    for index, agent_key in enumerate(agent_keys, start=1):
        numbered_lines.append(f"{index}. {agent_key}")

    console.print(
        Panel("\n".join(numbered_lines), title="Available Agents", border_style="cyan")
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter agent number[/bold]").strip()
        if not raw_choice:
            continue
        if not raw_choice.isdigit():
            console.print(Panel("Please enter a valid number.", border_style="yellow"))
            continue

        selected_index = int(raw_choice)
        if selected_index < 1 or selected_index > len(agent_keys):
            console.print(Panel("Agent number out of range.", border_style="yellow"))
            continue

        return agent_keys[selected_index - 1]


def select_verbosity_dialog(console: Console) -> int:
    console.print(
        Panel(
            "0. Minimal - shows one short line per tool call; does not show full I/O or reasoning\n"
            "1. Standard - shows full tool inputs and outputs; does not show reasoning\n"
            "3. Debug - shows full tool inputs/outputs and model reasoning (analysis/plan)",
            title="Verbosity Levels",
            border_style="cyan",
        )
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter verbosity (0, 1, or 3)[/bold]").strip()
        if not raw_choice:
            continue
        try:
            return _parse_verbosity(value=raw_choice)
        except ValueError:
            console.print(Panel("Please enter 0, 1, or 3.", border_style="yellow"))


def _interactive_help_panel() -> Panel:
    return Panel(
        "/model - choose active model from a numbered list\n"
        "/agent [name] - choose active agent or set by key\n"
        "/verbosity <0|1|3> - set output detail level\n"
        "  0: one line per tool call\n"
        "  1: full tool call inputs/responses\n"
        "  3: full inputs/responses + reasoning\n"
        "/max_turns <int>=1 - set max agent turns\n"
        "/max_wait_seconds <float>>0 - set per-command max wait",
        title="Interactive Commands",
        border_style="cyan",
    )


def parse_model_command(
    console: Console,
    instruction: str,
    config: LoadedConfig,
) -> tuple[str, str | None]:
    trimmed = instruction.strip()
    if trimmed != "/model" and not trimmed.startswith("/model "):
        return instruction, None

    remainder = trimmed.removeprefix("/model").strip()
    selected_model = select_model_dialog(console=console, config=config)
    return remainder, selected_model


def _parse_verbosity(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError("Verbosity must be one of: 0, 1, 3.") from err

    if parsed not in {0, 1, 3}:
        raise ValueError("Verbosity must be one of: 0, 1, 3.")

    return parsed


def _parse_max_turns(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError("max_turns must be an integer >= 1.") from err

    if parsed < 1:
        raise ValueError("max_turns must be an integer >= 1.")

    return parsed


def _parse_max_wait_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as err:
        raise ValueError("max_wait_seconds must be a number greater than 0.") from err

    if parsed <= 0:
        raise ValueError("max_wait_seconds must be a number greater than 0.")

    return parsed


def parse_interactive_command(
    console: Console,
    instruction: str,
    config: LoadedConfig,
) -> InteractiveCommandResult:
    trimmed = instruction.strip()
    if not trimmed:
        console.print(_interactive_help_panel())
        return InteractiveCommandResult(instruction="", handled=True)

    if not trimmed.startswith("/"):
        return InteractiveCommandResult(instruction=instruction, handled=False)

    if trimmed == "/model" or trimmed.startswith("/model "):
        remainder, selected_model = parse_model_command(
            console=console,
            instruction=instruction,
            config=config,
        )
        return InteractiveCommandResult(
            instruction=remainder,
            selected_model=selected_model,
            handled=True,
        )

    if trimmed == "/verbosity" or trimmed.startswith("/verbosity "):
        remainder = trimmed.removeprefix("/verbosity").strip()
        try:
            verbosity = (
                _parse_verbosity(value=remainder)
                if remainder
                else select_verbosity_dialog(console=console)
            )
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(Panel(f"Verbosity set to {verbosity}.", border_style="cyan"))
        return InteractiveCommandResult(
            instruction="",
            updated_verbosity=verbosity,
            handled=True,
        )

    if trimmed == "/agent" or trimmed.startswith("/agent "):
        remainder = trimmed.removeprefix("/agent").strip()
        if not remainder:
            remainder = select_agent_dialog(console=console, config=config)

        if remainder not in config.agents:
            console.print(
                Panel(
                    f"Unknown agent '{remainder}'. Available: {', '.join(config.agents.keys())}",
                    title="Invalid Command",
                    border_style="yellow",
                )
            )
            return InteractiveCommandResult(instruction="", handled=True)

        instruction_without_agent = ""
        console.print(Panel(f"Agent set to {remainder}.", border_style="cyan"))
        return InteractiveCommandResult(
            instruction=instruction_without_agent,
            selected_model=None,
            selected_agent=remainder,
            updated_verbosity=None,
            handled=True,
        )

    if trimmed == "/max_turns" or trimmed.startswith("/max_turns "):
        remainder = trimmed.removeprefix("/max_turns").strip()
        raw_value = (
            remainder or Prompt.ask("[bold]Enter max_turns (>= 1)[/bold]").strip()
        )
        try:
            config.max_turns = _parse_max_turns(value=raw_value)
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(
            Panel(f"max_turns set to {config.max_turns}.", border_style="cyan")
        )
        return InteractiveCommandResult(instruction="", handled=True)

    if trimmed == "/max_wait_seconds" or trimmed.startswith("/max_wait_seconds "):
        remainder = trimmed.removeprefix("/max_wait_seconds").strip()
        raw_value = (
            remainder or Prompt.ask("[bold]Enter max_wait_seconds (> 0)[/bold]").strip()
        )
        try:
            config.max_wait_seconds = _parse_max_wait_seconds(value=raw_value)
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(
            Panel(
                f"max_wait_seconds set to {config.max_wait_seconds}.",
                border_style="cyan",
            )
        )
        return InteractiveCommandResult(instruction="", handled=True)

    console.print(
        Panel(
            "Unknown command. Available commands: /model, /agent, /verbosity, /max_turns, /max_wait_seconds",
            title="Invalid Command",
            border_style="yellow",
        )
    )
    return InteractiveCommandResult(instruction="", handled=True)


def resolve_model_key(
    config: LoadedConfig,
    cli_model_key: str | None,
    selected_model_key: str | None,
) -> str:
    if cli_model_key:
        cleaned = cli_model_key.strip()
        if cleaned not in config.models:
            available = ", ".join(config.models.keys())
            raise ValueError(
                f"Unknown model key '{cleaned}'. Available model keys: {available}"
            )
        return cleaned

    if selected_model_key:
        return selected_model_key

    return config.default_model


def resolve_agent_key(
    config: LoadedConfig,
    cli_agent_key: str | None,
    selected_agent_key: str | None,
) -> str:
    if cli_agent_key:
        cleaned = cli_agent_key.strip()
        if cleaned not in config.agents:
            available = ", ".join(config.agents.keys())
            raise ValueError(
                f"Unknown agent key '{cleaned}'. Available agent keys: {available}"
            )
        return cleaned

    if selected_agent_key:
        return selected_agent_key

    return config.default_agent


def build_runtime_config(
    config: LoadedConfig,
    agent_key: str,
    model_key: str,
) -> AgentRuntimeConfig:
    selected = config.models[model_key]
    resolved_api_key = resolve_api_key(config_api_key=selected.api_key)
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


def main() -> None:
    console = Console()
    args = parse_args(sys.argv[1:])

    try:
        loaded_config = load_config(args.config)
    except Exception as err:
        console.print(Panel(str(err), title="Config Error", border_style="red"))
        raise SystemExit(1) from err

    if args.max_turns is not None:
        loaded_config.max_turns = max(1, args.max_turns)

    if args.verbosity is None:
        args.verbosity = loaded_config.verbosity

    instruction = " ".join(args.instruction).strip()
    selected_model_from_instruction: str | None = None
    selected_agent_from_instruction: str | None = None
    if not instruction:
        while True:
            candidate_instruction = Prompt.ask("[bold]Enter instruction[/bold]").strip()
            command_result = parse_interactive_command(
                console=console,
                instruction=candidate_instruction,
                config=loaded_config,
            )
            if command_result.handled:
                if command_result.selected_model:
                    selected_model_from_instruction = command_result.selected_model
                if command_result.selected_agent:
                    selected_agent_from_instruction = command_result.selected_agent
                if command_result.updated_verbosity is not None:
                    args.verbosity = command_result.updated_verbosity
                if command_result.instruction:
                    instruction = command_result.instruction
                    break
                continue

            if command_result.instruction:
                instruction = command_result.instruction
                break
    else:
        instruction, selected_model = parse_model_command(
            console=console,
            instruction=instruction,
            config=loaded_config,
        )
        if selected_model:
            selected_model_from_instruction = selected_model

    if not instruction:
        console.print(Panel("Instruction is required.", border_style="red"))
        raise SystemExit(1)

    try:
        active_agent_key = resolve_agent_key(
            config=loaded_config,
            cli_agent_key=args.agent,
            selected_agent_key=selected_agent_from_instruction,
        )
        active_model_key = resolve_model_key(
            config=loaded_config,
            cli_model_key=args.model,
            selected_model_key=selected_model_from_instruction,
        )
    except ValueError as err:
        console.print(Panel(str(err), title="Config Error", border_style="red"))
        raise SystemExit(1) from err

    model_entry = loaded_config.models[active_model_key]
    api_key = resolve_api_key(config_api_key=model_entry.api_key)
    if not api_key:
        env_name = _env_var_name(config_api_key=model_entry.api_key)
        if env_name:
            message = (
                f"API key not found for model '{active_model_key}'. "
                f"Set env var {env_name} or provide a literal api_key."
            )
        else:
            message = (
                f"API key not found for model '{active_model_key}'. "
                "Set api_key to a literal value or env var name."
            )
        console.print(
            Panel(
                message,
                title="Missing API Key",
                border_style="red",
            )
        )
        raise SystemExit(1)

    available = available_agents()
    if active_agent_key not in available:
        supported = ", ".join(sorted(available.keys()))
        console.print(
            Panel(
                f"Agent '{active_agent_key}' is not implemented. Supported agents: {supported}",
                title="Config Error",
                border_style="red",
            )
        )
        raise SystemExit(1)

    try:
        runtime_cfg = build_runtime_config(
            config=loaded_config,
            agent_key=active_agent_key,
            model_key=active_model_key,
        )
    except ValueError:
        console.print(
            Panel(
                f"API key not found for model '{active_model_key}'.",
                title="Missing API Key",
                border_style="red",
            )
        )
        raise SystemExit(1) from None

    runtime_cfg.agent_config["verbosity"] = args.verbosity

    cwd = os.getcwd()

    if args.verbosity == 0:
        panel_lines = [
            f"Agent: {active_agent_key}",
            f"Model: {active_model_key}",
            f"CWD: {cwd}",
        ]
    else:
        panel_lines = [
            f"Agent: {active_agent_key}",
            f"Model Key: {active_model_key}",
            f"Model: {runtime_cfg.model.model}",
            f"API Base: {runtime_cfg.model.api_base}",
            f"CWD: {cwd}",
            f"Verbosity: {args.verbosity}",
            f"Max Turns: {runtime_cfg.agent_config['max_turns']}",
            f"Max Wait: {runtime_cfg.agent_config['max_wait_seconds']}s",
        ]

    console.print(
        Panel(
            "\n".join(panel_lines),
            title="Multi-Agent Wrapper",
            border_style="cyan",
        )
    )

    agent = get_agent(active_agent_key)
    raise SystemExit(
        agent.run(
            instruction=instruction,
            cfg=runtime_cfg,
            console=console,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print(
            Panel("Cancelled by user.", title="Stopped", border_style="yellow")
        )
        raise SystemExit(130)
