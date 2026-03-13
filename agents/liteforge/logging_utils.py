from __future__ import annotations

from rich.console import Console
from rich.text import Text

from agents.liteforge.context import Context


def display_width(*, console: Console) -> int:
    return max(20, console.width)


def print_rule(*, console: Console) -> None:
    console.print(Text("─" * display_width(console=console), style="dim"))


def last_assistant_text(*, context: Context) -> str | None:
    for message in reversed(context.messages):
        if message.role == "assistant" and message.content:
            content = message.content.strip()
            if content:
                return content

    return None
