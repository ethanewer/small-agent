# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportUnusedVariable=false

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text

from agents.liteforge.context import Context, ToolCall, ToolResult
from agents.liteforge.logging_utils import print_rule
from agents.liteforge.provider import chat
from agents.liteforge.tools.executor import ToolExecutor


@dataclass
class ToolErrorTracker:
    max_failures: int = 3
    _failure_counts: dict[str, int] = field(default_factory=dict)

    def record_failure(self, tool_name: str) -> None:
        self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1

    def record_success(self, tool_name: str) -> None:
        self._failure_counts.pop(tool_name, None)

    def remaining_attempts(self, tool_name: str) -> int:
        used = self._failure_counts.get(tool_name, 0)
        return max(0, self.max_failures - used)

    def total_failures(self) -> int:
        return sum(self._failure_counts.values())

    def limit_reached(self) -> bool:
        return self.total_failures() >= self.max_failures

    def errors(self) -> dict[str, int]:
        return dict(self._failure_counts)


class Orchestrator:
    """Core agent loop matching orch.rs."""

    def __init__(
        self,
        context: Context,
        executor: ToolExecutor,
        model: str,
        tools: list[dict[str, object]],
        max_turns: int = 100,
        max_tool_failure_per_turn: int = 3,
        stream: bool = True,
    ) -> None:
        self.context = context
        self.executor = executor
        self.model = model
        self.tools = tools
        self.max_turns = max_turns
        self.error_tracker = ToolErrorTracker(max_failures=max_tool_failure_per_turn)
        self.stream = stream
        self._log_console = Console(stderr=True)
        self._stream_line_open = False
        self._stream_had_visible_text_since_break = False
        self._pending_trailing_newlines = ""
        self._pending_stream_separator = False
        self._streamed_text = ""
        self._visible_text = ""

    def set_log_console(self, *, console: Console | None) -> None:
        if console is None:
            return

        self._log_console = console

    def queue_stream_separator(self) -> None:
        self._pending_stream_separator = True

    def _print_rule(self) -> None:
        print_rule(console=self._log_console)

    def _ensure_stream_line_break(self) -> None:
        if self._pending_trailing_newlines:
            # Only emit a separator newline when streamed content actually printed
            # visible characters before trailing newline chunks.
            if self._stream_had_visible_text_since_break:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._pending_trailing_newlines = ""
            self._stream_line_open = False
            self._stream_had_visible_text_since_break = False
            return

        if not self._stream_line_open:
            return

        sys.stdout.write("\n")
        sys.stdout.flush()
        self._stream_line_open = False
        self._stream_had_visible_text_since_break = False

    def _render_tool_log(self, *, tc: ToolCall) -> None:
        mapping: dict[str, tuple[str, str, str]] = {
            "shell": ("shell", "cyan", tc.arguments.get("description", "")),
            "read": ("read", "blue", tc.arguments.get("file_path", "")),
            "write": ("write", "yellow", tc.arguments.get("file_path", "")),
            "patch": ("patch", "yellow", tc.arguments.get("file_path", "")),
            "fs_search": ("search", "magenta", tc.arguments.get("pattern", "")),
            "remove": ("remove", "red", tc.arguments.get("path", "")),
            "fetch": ("fetch", "green", tc.arguments.get("url", "")),
            "todo_write": ("todo_write", "cyan", ""),
            "todo_read": ("todo_read", "cyan", ""),
        }
        display_name: str
        color: str
        detail: str
        if tc.name in mapping:
            display_name, color, detail = mapping[tc.name]
            if tc.name == "shell" and not detail:
                detail = str(tc.arguments.get("command", ""))
        else:
            display_name = tc.name
            color = "cyan"
            detail = ""
        detail = detail.strip()

        self._ensure_stream_line_break()
        self._print_rule()
        line = Text("[", style="dim")
        line.append(display_name, style=f"bold {color}")
        line.append("]", style="dim")
        if detail:
            line.append(" ")
            line.append(detail, style="white")
        self._log_console.print(line)
        self._pending_stream_separator = True

    def _render_status(self, *, label: str, message: str, color: str) -> None:
        self._ensure_stream_line_break()
        self._print_rule()
        line = Text(f"{label}: ", style=f"bold {color}")
        line.append(message, style="white")
        self._log_console.print(line)

    def _render_todo_state(self, *, state_text: str, is_error: bool) -> None:
        style = "red" if is_error else "white"
        if not state_text.strip():
            self._log_console.print("[empty]", style="dim")
            return

        self._visible_text += f"{state_text}\n"
        self._log_console.print(state_text, style=style, markup=False)

    def _stream_callback(self, text: str) -> None:
        if not text:
            return

        self._streamed_text += text
        self._visible_text += text
        full_text = self._pending_trailing_newlines + text
        self._pending_trailing_newlines = ""
        if self._pending_stream_separator and full_text.strip():
            self._print_rule()
            self._pending_stream_separator = False

        trimmed = full_text.rstrip("\n")
        trailing = full_text[len(trimmed) :]
        if trimmed:
            sys.stdout.write(trimmed)
            sys.stdout.flush()
            self._stream_line_open = True
            self._stream_had_visible_text_since_break = True

        if trailing:
            self._pending_trailing_newlines = trailing
            self._stream_line_open = False

    def run(self) -> bool:
        """Execute the main agent loop."""
        should_yield = False
        is_complete = False
        turn_count = 0
        failed = False

        while not should_yield:
            callback = self._stream_callback if self.stream else None

            try:
                response = chat(
                    self.context,
                    self.model,
                    self.tools,
                    stream_callback=callback,
                )
            except Exception as e:
                self._render_status(
                    label="error", message=f"Error calling LLM: {e}", color="red"
                )
                failed = True
                break

            if self.stream and response.content:
                pass

            is_complete = response.finish_reason == "stop" and not response.tool_calls

            should_yield = is_complete

            tool_call_records = self._execute_tool_calls(response.tool_calls)

            for tc, result in tool_call_records:
                if result.is_error:
                    self.error_tracker.record_failure(result.name)
                    attempts_left = self.error_tracker.remaining_attempts(result.name)
                    retry_msg = (
                        f"\n<retry>\nTool call failed\n"
                        f"- **Attempts remaining:** {attempts_left}\n"
                        f"- **Next steps:** Analyze the error, identify the root cause, "
                        f"and adjust your approach before retrying.\n</retry>"
                    )
                    result.content += retry_msg
                else:
                    self.error_tracker.record_success(result.name)

            self.context.append_turn(
                response.content,
                response.tool_calls if response.tool_calls else None,
                tool_call_records,
            )

            if self.error_tracker.limit_reached():
                self._render_status(
                    label="error",
                    message=(
                        "Max tool failure limit reached "
                        f"({self.error_tracker.max_failures}). Errors: "
                        f"{self.error_tracker.errors()}"
                    ),
                    color="red",
                )
                should_yield = True
                failed = True

            turn_count += 1
            if not should_yield and turn_count >= self.max_turns:
                self._render_status(
                    label="error",
                    message=f"Reached max turns ({self.max_turns}).",
                    color="red",
                )
                should_yield = True
                failed = True

        if is_complete and self.stream:
            self._ensure_stream_line_break()

        return not failed

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[tuple[ToolCall, ToolResult]]:
        records = []
        for tc in tool_calls:
            if self.stream:
                self._render_tool_log(tc=tc)

            output, is_error = self.executor.execute(tc.name, tc.arguments)
            if self.stream and tc.name in {"todo_write", "todo_read"}:
                self._render_todo_state(state_text=output, is_error=is_error)

            result = ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=output,
                is_error=is_error,
            )
            records.append((tc, result))

        return records

    @property
    def streamed_text(self) -> str:
        return self._streamed_text

    @property
    def visible_text(self) -> str:
        return self._visible_text
