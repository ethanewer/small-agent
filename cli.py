from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from agents.core.task import Task
from agents.interface import run_agent_task_with_fallback
from agents.registry import available_agents, get_agent
from harbor_config import (
    CONFIG_PATH,
    ConfigModelEntry,  # noqa: F401 -- re-exported for tests
    LoadedConfig,
    _env_var_name,
    build_runtime_config,
    load_config,
    resolve_api_key,
)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


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
        choices=[0, 1],
        default=None,
        help="0: one line per tool call, 1: full tool inputs/responses + reasoning",
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
    parser.add_argument(
        "--no-final-message",
        action="store_true",
        default=False,
        help="Disable the final summary message (used by benchmarks).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Enable planning mode (read-only workflow for liteforge agent).",
    )

    return parser.parse_args(argv)


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
            "1. Full - shows full tool inputs/outputs and model reasoning (analysis/plan)",
            title="Verbosity Levels",
            border_style="cyan",
        )
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter verbosity (0 or 1)[/bold]").strip()
        if not raw_choice:
            continue

        try:
            return _parse_verbosity(value=raw_choice)
        except ValueError:
            console.print(Panel("Please enter 0 or 1.", border_style="yellow"))


def _interactive_help_panel() -> Panel:
    return Panel(
        "/model - choose active model from a numbered list\n"
        "/agent [name] - choose active agent or set by key\n"
        "/verbosity <0|1> - set output detail level\n"
        "  0: one line per tool call\n"
        "  1: full inputs/responses + reasoning\n"
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
        raise ValueError("Verbosity must be one of: 0, 1.") from err

    if parsed not in {0, 1}:
        raise ValueError("Verbosity must be one of: 0, 1.")

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
    agent_key: str | None = None,
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

    if agent_key:
        agent_cfg = config.agents.get(agent_key, {})
        agent_default = agent_cfg.get("default_model")
        if isinstance(agent_default, str) and agent_default.strip() in config.models:
            return agent_default.strip()

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
            agent_key=active_agent_key,
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
    if args.no_final_message:
        runtime_cfg.agent_config["final_message"] = False
    if args.plan:
        runtime_cfg.agent_config["plan_mode"] = True
        runtime_cfg.agent_config["readonly"] = True

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
        if args.plan:
            panel_lines.append("Plan Mode: enabled")

    console.print(
        Panel(
            "\n".join(panel_lines),
            title="Multi-Agent Wrapper",
            border_style="cyan",
        )
    )

    agent = get_agent(active_agent_key)
    task = Task.from_instruction(instruction=instruction, task_id="cli-interactive")
    result = run_agent_task_with_fallback(
        agent=agent,
        task=task,
        cfg=runtime_cfg,
        console=console,
        sink=None,
    )
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print(
            Panel("Cancelled by user.", title="Stopped", border_style="yellow")
        )
        raise SystemExit(130)
