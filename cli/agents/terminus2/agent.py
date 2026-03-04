from __future__ import annotations

import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.core.events import AgentEvent
from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig
from agents.terminus2.core_agent import (
    AgentCallbacks,
    Command,
    Config as CoreConfig,
    ModelConfig as CoreModelConfig,
    ParsedResponse,
    run_agent,
)


def _coerce_final_message_enabled(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _display_width(console: Console) -> int:
    return max(20, console.width)


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
    content_width = max(10, width - prefix_len)
    if len(text) > content_width:
        return text[: content_width - 3] + "..."
    return text.ljust(content_width)


def _render_response(
    console: Console,
    turn: int,
    parsed: ParsedResponse,
    verbosity: int,
) -> None:
    if verbosity >= 3:
        reasoning = f"analysis:\n{parsed.analysis}\n\nplan:\n{parsed.plan}"
        console.print(
            Panel(reasoning, title=f"Turn {turn} Reasoning", border_style="magenta")
        )


def _render_command_output(
    console: Console,
    command: Command,
    output: str,
    verbosity: int,
) -> None:
    width = _display_width(console)
    if command.keystrokes == "":
        input_text = "<wait>"
    elif command.keystrokes.strip() == "":
        input_text = "<enter>"
    else:
        input_text = command.keystrokes

    display_input = input_text.replace("\n", "\\n")
    normalized_output = output.strip() if output else ""
    output_text = normalized_output if normalized_output else "[no output]"
    if verbosity == 0:
        in_prefix = "in: "
        out_prefix = "out: "
        preview = display_input
        response_preview = output_text.replace("\n", " ")
        console.print(Text("─" * width, style="dim"))
        in_line = Text(in_prefix, style="cyan")
        in_line.append(
            _fit_line(
                text=preview or "<wait>",
                width=width,
                prefix_len=len(in_prefix),
            ),
            style="white",
        )
        console.print(in_line)
        out_line = Text(out_prefix, style="green")
        out_line.append(
            _fit_line(
                text=response_preview,
                width=width,
                prefix_len=len(out_prefix),
            ),
            style="white",
        )
        console.print(out_line)
        return

    console.print(Text("─" * width, style="dim"))
    _render_labeled_fixed(
        console=console,
        width=width,
        label="cmd: ",
        label_style="cyan",
        content=display_input,
    )
    _render_labeled_fixed(
        console=console,
        width=width,
        label="out: ",
        label_style="green",
        content=output_text,
    )


def _render_issue_output(
    console: Console,
    kind: str,
    message: str,
    verbosity: int,
) -> None:
    if verbosity == 0 and kind != "model":
        return

    width = _display_width(console)
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

    console.print(Text("─" * width, style="dim"))
    error_line = Text("error: ", style="red")
    error_line.append(kind, style="white")
    error_line.append(" " * max(0, width - len("error: ") - len(kind)), style="white")
    console.print(error_line)
    for idx, segment in enumerate(wrapped):
        prefix = "details: " if idx == 0 else (" " * len("details: "))
        line = Text(prefix, style="red")
        line.append(segment.ljust(content_width), style="white")
        console.print(line)


class Terminus2Agent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        result = self.run_task(
            task=Task.from_instruction(instruction=instruction),
            cfg=cfg,
            console=console,
            sink=None,
        )
        return result.exit_code

    def run_task(
        self,
        *,
        task: Task,
        cfg: AgentRuntimeConfig,
        console: Console | None = None,
        sink: EventSink | None = None,
    ) -> RunResult:
        if console is None:
            console = Console()
        verbosity = int(cfg.agent_config.get("verbosity", 1))
        max_turns = int(cfg.agent_config.get("max_turns", 50))
        max_wait_seconds = float(cfg.agent_config.get("max_wait_seconds", 60.0))
        final_message_enabled = _coerce_final_message_enabled(
            cfg.agent_config.get("final_message")
        )
        core_cfg = CoreConfig(
            active_model_key=cfg.model.model,
            active_model=CoreModelConfig(
                model=cfg.model.model,
                api_base=cfg.model.api_base,
                api_key=cfg.model.api_key,
                temperature=cfg.model.temperature,
            ),
            verbosity=verbosity,
            max_turns=max_turns,
            max_wait_seconds=max_wait_seconds,
            final_message_enabled=final_message_enabled,
        )

        def on_reasoning(turn: int, parsed: ParsedResponse) -> None:
            _render_response(
                console=console,
                turn=turn,
                parsed=parsed,
                verbosity=verbosity,
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="reasoning",
                        turn=turn,
                        payload={"analysis": parsed.analysis, "plan": parsed.plan},
                    )
                )

        def on_command_output(command: Command, output: str) -> None:
            _render_command_output(
                console=console,
                command=command,
                output=output,
                verbosity=verbosity,
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="command_output",
                        payload={
                            "keystrokes": command.keystrokes,
                            "duration": command.duration,
                            "output": output,
                        },
                    )
                )

        def on_issue(kind: str, message: str) -> None:
            _render_issue_output(
                console=console,
                kind=kind,
                message=message,
                verbosity=verbosity,
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={"kind": kind, "message": message},
                    )
                )

        def on_done(done_text: str) -> None:
            console.print(Panel(done_text, title="Done", border_style="green"))
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="done",
                        payload={"message": done_text},
                    )
                )

        def on_stopped(stopped_max_turns: int) -> None:
            console.print(
                Panel(
                    f"Reached max turns ({stopped_max_turns}) without completion.",
                    title="Stopped",
                    border_style="yellow",
                )
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="stopped",
                        payload={"max_turns": stopped_max_turns},
                    )
                )

        callbacks = AgentCallbacks(
            on_reasoning=on_reasoning,
            on_command_output=on_command_output,
            on_issue=on_issue,
            on_done=on_done,
            on_stopped=on_stopped,
        )
        exit_code = run_agent(
            instruction=task.instruction,
            cfg=core_cfg,
            api_key=cfg.model.api_key,
            callbacks=callbacks,
        )
        result = RunResult(
            exit_code=exit_code,
            success=exit_code == 0,
            task_id=task.task_id,
        )
        if sink:
            sink.finalize(result=result)

        return result
