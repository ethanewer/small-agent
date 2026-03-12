from __future__ import annotations

import sys
from dataclasses import dataclass, field

from agents.liteforge.context import Context, ToolCall, ToolResult
from agents.liteforge.provider import chat
from agents.liteforge.tools.executor import ToolExecutor
from agents.liteforge.tools.registry import YIELD_TOOLS


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
        tools: list[dict],
        max_requests_per_turn: int = 100,
        max_tool_failure_per_turn: int = 3,
        stream: bool = True,
    ) -> None:
        self.context = context
        self.executor = executor
        self.model = model
        self.tools = tools
        self.max_requests_per_turn = max_requests_per_turn
        self.error_tracker = ToolErrorTracker(max_failures=max_tool_failure_per_turn)
        self.stream = stream
        self._last_plan_content: str | None = None

    def _stream_callback(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def run(self) -> bool:
        """Execute the main agent loop."""
        should_yield = False
        is_complete = False
        request_count = 0
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
                print(f"\nError calling LLM: {e}", file=sys.stderr)
                failed = True
                break

            if self.stream and response.content:
                pass

            is_complete = response.finish_reason == "stop" and not response.tool_calls

            should_yield = is_complete or any(
                tc.name in YIELD_TOOLS for tc in response.tool_calls
            )

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

                if tc.name == "plan" and not result.is_error:
                    self._last_plan_content = result.content

            self.context.append_turn(
                response.content,
                response.tool_calls if response.tool_calls else None,
                tool_call_records,
            )

            if self.error_tracker.limit_reached():
                print(
                    f"\nMax tool failure limit reached ({self.error_tracker.max_failures}). "
                    f"Errors: {self.error_tracker.errors()}",
                    file=sys.stderr,
                )
                should_yield = True
                failed = True

            request_count += 1
            if not should_yield and request_count >= self.max_requests_per_turn:
                print(
                    f"\nMax requests per turn limit reached ({self.max_requests_per_turn}).",
                    file=sys.stderr,
                )
                should_yield = True
                failed = True

        if is_complete and self.stream:
            print()

        return not failed

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[tuple[ToolCall, ToolResult]]:
        records = []
        for tc in tool_calls:
            if self.stream:
                desc = tc.arguments.get("description", "")
                if tc.name == "shell":
                    cmd = tc.arguments.get("command", "")
                    print(f"\n[shell] {desc or cmd}", file=sys.stderr)
                elif tc.name == "read":
                    path = tc.arguments.get("file_path", "")
                    print(f"\n[read] {path}", file=sys.stderr)
                elif tc.name == "write":
                    path = tc.arguments.get("file_path", "")
                    print(f"\n[write] {path}", file=sys.stderr)
                elif tc.name == "patch":
                    path = tc.arguments.get("file_path", "")
                    print(f"\n[patch] {path}", file=sys.stderr)
                elif tc.name == "fs_search":
                    pattern = tc.arguments.get("pattern", "")
                    print(f"\n[search] {pattern}", file=sys.stderr)
                elif tc.name == "remove":
                    path = tc.arguments.get("path", "")
                    print(f"\n[remove] {path}", file=sys.stderr)
                elif tc.name == "fetch":
                    url = tc.arguments.get("url", "")
                    print(f"\n[fetch] {url}", file=sys.stderr)
                elif tc.name in ("todo_write", "todo_read"):
                    print(f"\n[{tc.name}]", file=sys.stderr)
                elif tc.name == "plan":
                    name = tc.arguments.get("plan_name", "")
                    print(f"\n[plan] {name}", file=sys.stderr)
                elif tc.name == "followup":
                    pass
                else:
                    print(f"\n[{tc.name}]", file=sys.stderr)

            output, is_error = self.executor.execute(tc.name, tc.arguments)

            result = ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=output,
                is_error=is_error,
            )
            records.append((tc, result))

        return records

    @property
    def last_plan_content(self) -> str | None:
        return self._last_plan_content
