from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import textwrap
from typing import Any

from agents import run
from harbor_config import (
    CONFIG_PATH,
    ConfigModelEntry,  # noqa: F401 -- re-exported for tests
    LoadedConfig,
    _env_var_name,
    build_config,
    load_config,
    resolve_api_key,
)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text


class ConsoleLogger:
    def __init__(self, *, console: Console, verbosity: int) -> None:
        self._console = console
        self._verbosity = verbosity
        self._compaction_counts: dict[str, int] = {"proactive": 0, "reactive": 0}

    def log(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        turn: int | None = None,
    ) -> None:
        if event_type == "reasoning":
            self._render_reasoning(turn=turn or 0, payload=payload)
        elif event_type == "command_output":
            self._render_command_output(payload=payload)
        elif event_type == "issue":
            self._render_issue(payload=payload)
        elif event_type == "done":
            self._console.print(
                Panel(payload["message"], title="Done", border_style="green")
            )
        elif event_type == "stopped":
            self._console.print(
                Panel(
                    f"Reached max turns ({payload['max_turns']}) without completion.",
                    title="Stopped",
                    border_style="yellow",
                )
            )
        elif event_type == "compaction":
            kind = payload.get("kind", "unknown")
            self._compaction_counts[kind] = self._compaction_counts.get(kind, 0) + 1

    def print_compaction_summary(self) -> None:
        total = self._compaction_counts.get(
            "proactive", 0
        ) + self._compaction_counts.get("reactive", 0)
        self._console.print(
            f"Compactions: {total} "
            f"(proactive={self._compaction_counts.get('proactive', 0)}, "
            f"reactive={self._compaction_counts.get('reactive', 0)})"
        )

    def _render_reasoning(self, turn: int, payload: dict[str, Any]) -> None:
        if self._verbosity >= 1:
            reasoning = f"analysis:\n{payload['analysis']}\n\nplan:\n{payload['plan']}"
            self._console.print(
                Panel(reasoning, title=f"Turn {turn} Reasoning", border_style="magenta")
            )

    def _render_command_output(self, payload: dict[str, Any]) -> None:
        width = self._display_width()
        keystrokes = payload["keystrokes"]
        output = payload["output"]

        if keystrokes == "":
            input_text = "<wait>"
        elif keystrokes.strip() == "":
            input_text = "<enter>"
        else:
            input_text = keystrokes

        display_input = input_text.replace("\n", "\\n")
        raw = output
        for prefix in ("New Terminal Output:\n", "Current Terminal Screen:\n"):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]
                break

        normalized_output = raw.strip() if raw else ""
        output_text = normalized_output if normalized_output else "[no output]"

        if self._verbosity == 0:
            in_prefix = "in: "
            out_prefix = "out: "
            response_preview = output_text.replace("\n", " ")
            self._console.print(Text("─" * width, style="dim"))
            in_line = Text(in_prefix, style="cyan")
            in_line.append(
                _fit_line(
                    text=display_input or "<wait>",
                    width=width,
                    prefix_len=len(in_prefix),
                ),
                style="white",
            )
            self._console.print(in_line)
            out_line = Text(out_prefix, style="green")
            out_line.append(
                _fit_line(
                    text=response_preview,
                    width=width,
                    prefix_len=len(out_prefix),
                ),
                style="white",
            )
            self._console.print(out_line)
            return

        self._console.print(Text("─" * width, style="dim"))
        _render_labeled_fixed(
            console=self._console,
            width=width,
            label="cmd: ",
            label_style="cyan",
            content=display_input,
        )
        _render_labeled_fixed(
            console=self._console,
            width=width,
            label="out: ",
            label_style="green",
            content=output_text,
        )

    def _render_issue(self, payload: dict[str, Any]) -> None:
        kind = payload["kind"]
        message = payload["message"]

        if self._verbosity == 0 and kind != "model":
            return

        width = self._display_width()
        content_width = max(10, width - len("details: "))
        details_text = message.replace("\n", " ")
        wrapped = textwrap.wrap(
            details_text,
            width=content_width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            wrapped = [""]

        self._console.print(Text("─" * width, style="dim"))
        error_line = Text("error: ", style="red")
        error_line.append(kind, style="white")
        error_line.append(
            " " * max(0, width - len("error: ") - len(kind)), style="white"
        )
        self._console.print(error_line)
        for idx, segment in enumerate(wrapped):
            prefix = "details: " if idx == 0 else (" " * len("details: "))
            line = Text(prefix, style="red")
            line.append(segment.ljust(content_width), style="white")
            self._console.print(line)

    def _display_width(self) -> int:
        return max(20, self._console.width)


def _render_labeled_fixed(
    console: Console,
    width: int,
    label: str,
    label_style: str,
    content: str,
) -> None:
    content_width = max(10, width - len(label))
    lines = content.splitlines() or [""]
    first = True
    for raw_line in lines:
        wrapped = textwrap.wrap(
            raw_line,
            width=content_width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            wrapped = [""]

        for segment in wrapped:
            prefix = label if first else (" " * len(label))
            line = Text(prefix, style=label_style)
            line.append(segment.ljust(content_width), style="white")
            console.print(line)
            first = False


def _fit_line(text: str, width: int, prefix_len: int) -> str:
    max_chars = width - prefix_len
    if max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…"


@dataclass
class InteractiveCommandResult:
    instruction: str
    selected_model: str | None = None
    updated_verbosity: int | None = None
    handled: bool = False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal agent CLI")
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


def select_verbosity_dialog(console: Console) -> int:
    console.print(
        Panel(
            "0. Minimal - shows one short line per tool call; does not show full I/O or reasoning\n"
            "1. Full - shows full tool inputs/outputs and model reasoning",
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
            "Unknown command. Available commands: /model, /verbosity, /max_turns, /max_wait_seconds",
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

    try:
        config = build_config(
            loaded_config=loaded_config,
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

    cwd = os.getcwd()
    if args.verbosity == 0:
        panel_lines = [
            f"Model: {active_model_key}",
            f"CWD: {cwd}",
        ]
    else:
        panel_lines = [
            f"Model Key: {active_model_key}",
            f"Model: {config.model}",
            f"API Base: {config.api_base}",
            f"CWD: {cwd}",
            f"Verbosity: {args.verbosity}",
            f"Max Turns: {config.max_turns}",
            f"Max Wait: {config.max_wait_seconds}s",
        ]

    console.print(
        Panel(
            "\n".join(panel_lines),
            title="Terminal Agent",
            border_style="cyan",
        )
    )

    logger = ConsoleLogger(console=console, verbosity=args.verbosity)
    result = run(
        instruction=instruction,
        config=config,
        logger=logger,  # pyright: ignore[reportArgumentType]
    )
    logger.print_compaction_summary()
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print(
            Panel("Cancelled by user.", title="Stopped", border_style="yellow")
        )
        raise SystemExit(130)
