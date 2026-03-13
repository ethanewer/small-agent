from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.text import Text

from agents.liteforge.context import Context


PLAN_CREATED_PREFIX = "Plan created:"


@dataclass(frozen=True)
class PlanDisplayDecision:
    plan_text: str | None
    plan_source: str
    should_print: bool
    was_visible: bool


def display_width(*, console: Console) -> int:
    return max(20, console.width)


def print_rule(*, console: Console) -> None:
    console.print(Text("─" * display_width(console=console), style="dim"))


def print_status(*, console: Console, message: str, style: str = "bold cyan") -> None:
    print_rule(console=console)
    console.print(message, style=style)


def print_plain_text_block(*, console: Console, text: str) -> None:
    print_rule(console=console)
    console.print(text, markup=False)


def last_assistant_text(*, context: Context) -> str | None:
    for message in reversed(context.messages):
        if message.role == "assistant" and message.content:
            content = message.content.strip()
            if content:
                return content

    return None


def resolve_plan_text(*, orch: object) -> tuple[str | None, str]:
    plan_text = getattr(orch, "last_plan_content", None)
    if isinstance(plan_text, str) and plan_text.strip():
        return plan_text.strip(), "plan_tool"

    context = getattr(orch, "context", None)
    if isinstance(context, Context):
        for message in reversed(context.messages):
            if message.role != "tool" or not message.tool_result:
                continue

            if message.tool_result.name != "todo_write":
                continue

            todo_text = (message.tool_result.content or "").strip()
            if todo_text:
                return todo_text, "todo_write"

        assistant_text = last_assistant_text(context=context)
        if assistant_text:
            return assistant_text, "assistant_fallback"

    visible_text = _visible_text(orch=orch).strip()
    if visible_text:
        return visible_text, "visible_fallback"

    return None, "none"


def resolve_plan_display(
    *,
    orch: object,
    stream: bool,
    base_dir: Path | None = None,
) -> PlanDisplayDecision:
    plan_text, plan_source = resolve_plan_text(orch=orch)
    if plan_text is None:
        return PlanDisplayDecision(
            plan_text=None,
            plan_source=plan_source,
            should_print=False,
            was_visible=False,
        )

    materialized = materialize_plan_text(
        plan_text=plan_text,
        plan_source=plan_source,
        base_dir=base_dir,
    ).strip()
    if not materialized:
        return PlanDisplayDecision(
            plan_text=None,
            plan_source=plan_source,
            should_print=False,
            was_visible=False,
        )

    already_visible = stream and _contains_normalized(
        haystack=_visible_text(orch=orch),
        needle=materialized,
    )
    should_print = not already_visible
    return PlanDisplayDecision(
        plan_text=materialized,
        plan_source=plan_source,
        should_print=should_print,
        was_visible=already_visible or should_print,
    )


def materialize_plan_text(
    *,
    plan_text: str,
    plan_source: str,
    base_dir: Path | None = None,
) -> str:
    if plan_source != "plan_tool" and not _looks_like_plan_path(value=plan_text):
        return plan_text

    path_line = _extract_path_line(value=plan_text)
    if not path_line:
        return plan_text

    for path in _candidate_paths(path_line=path_line, base_dir=base_dir):
        try:
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if content:
            return content

    return plan_text


def _visible_text(*, orch: object) -> str:
    visible = getattr(orch, "visible_text", None)
    if isinstance(visible, str):
        return visible

    streamed = getattr(orch, "streamed_text", None)
    if isinstance(streamed, str):
        return streamed

    return ""


def _contains_normalized(*, haystack: str, needle: str) -> bool:
    normalized_needle = _normalize(value=needle)
    normalized_haystack = _normalize(value=haystack)
    if not normalized_needle or not normalized_haystack:
        return False

    return normalized_needle in normalized_haystack


def _normalize(*, value: str) -> str:
    return " ".join(value.split())


def _looks_like_plan_path(*, value: str) -> bool:
    return value.strip().lower().startswith(PLAN_CREATED_PREFIX.lower())


def _extract_path_line(*, value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None

    if not stripped.lower().startswith(PLAN_CREATED_PREFIX.lower()):
        return None

    remainder = stripped[len(PLAN_CREATED_PREFIX) :].strip()
    if not remainder:
        return None

    return remainder.splitlines()[0].strip()


def _candidate_paths(*, path_line: str, base_dir: Path | None) -> list[Path]:
    candidates: list[str] = [path_line]
    markdown_match = re.search(r"\]\(([^)]+)\)", path_line)
    if markdown_match:
        candidates.append(markdown_match.group(1))

    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        variants = [candidate, candidate.rstrip(".,;:!?)")]
        for variant in variants:
            normalized = _strip_wrapping(value=variant).strip()
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            raw_path = Path(normalized)
            if raw_path.is_absolute():
                paths.append(raw_path)
            elif base_dir is not None:
                paths.append(base_dir / raw_path)
            else:
                paths.append(raw_path)

    return paths


def _strip_wrapping(*, value: str) -> str:
    stripped = value.strip()
    wrappers = [("`", "`"), ("'", "'"), ('"', '"'), ("<", ">"), ("(", ")")]

    changed = True
    while changed and stripped:
        changed = False
        for left, right in wrappers:
            if stripped.startswith(left) and stripped.endswith(right):
                stripped = stripped[1:-1].strip()
                changed = True

    return stripped
