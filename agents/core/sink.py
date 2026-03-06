from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.panel import Panel

from agents.core.events import AgentEvent
from agents.core.result import RunResult


class EventSink(Protocol):
    def emit(self, *, event: AgentEvent) -> None: ...

    def finalize(self, *, result: RunResult) -> None: ...


class ConsoleEventSink:
    def __init__(self, *, console: Console, verbosity: int = 1) -> None:
        self._console = console
        self._verbosity = verbosity

    def emit(self, *, event: AgentEvent) -> None:
        if self._verbosity == 0 and event.event_type in {"reasoning", "tool_result"}:
            return

        self._console.print(
            Panel(
                str(event.payload),
                title=f"event: {event.event_type}",
                border_style="cyan",
            )
        )

    def finalize(self, *, result: RunResult) -> None:
        status = "success" if result.success else "failure"
        self._console.print(
            Panel(
                f"task_id={result.task_id}\nexit_code={result.exit_code}\nstatus={status}",
                title="Run Result",
                border_style="green" if result.success else "red",
            )
        )


class JsonlEventSink:
    def __init__(self, *, output_path: Path) -> None:
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, *, event: AgentEvent) -> None:
        record = {
            "event_type": event.event_type,
            "turn": event.turn,
            "payload": event.payload,
        }
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def finalize(self, *, result: RunResult) -> None:
        record = {
            "event_type": "result",
            "payload": {
                "task_id": result.task_id,
                "exit_code": result.exit_code,
                "success": result.success,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "trace_path": result.trace_path,
                "final_message": result.final_message,
            },
        }
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
