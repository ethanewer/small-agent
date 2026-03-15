# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

MAX_SEARCH_RESULT_BYTES = 10240


def _coerce_optional_int(*, value: Any, name: str) -> tuple[int | None, str | None]:
    if value is None or value == "":
        return None, None

    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, f"Error: {name} must be an integer"


def _coerce_bool(*, value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    return bool(value)


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    pattern = str(args.get("pattern", "")).strip()
    raw_search_path = args.get("path")
    search_path = str(raw_search_path).strip() if raw_search_path is not None else None
    raw_glob_pattern = args.get("glob")
    glob_pattern = (
        str(raw_glob_pattern).strip() if raw_glob_pattern is not None else None
    )
    output_mode = args.get("output_mode", "files_with_matches")
    before_ctx, before_ctx_error = _coerce_optional_int(
        value=args.get("-B"),
        name="-B",
    )
    if before_ctx_error:
        return before_ctx_error

    after_ctx, after_ctx_error = _coerce_optional_int(
        value=args.get("-A"),
        name="-A",
    )
    if after_ctx_error:
        return after_ctx_error

    context, context_error = _coerce_optional_int(
        value=args.get("-C"),
        name="-C",
    )
    if context_error:
        return context_error

    show_line_numbers = _coerce_bool(value=args.get("-n"))
    case_insensitive = _coerce_bool(value=args.get("-i"))

    raw_file_type = args.get("type")
    file_type = str(raw_file_type).strip() if raw_file_type is not None else None
    if file_type == "":
        file_type = None

    head_limit, head_limit_error = _coerce_optional_int(
        value=args.get("head_limit"),
        name="head_limit",
    )
    if head_limit_error:
        return head_limit_error

    offset, offset_error = _coerce_optional_int(
        value=args.get("offset"),
        name="offset",
    )
    if offset_error:
        return offset_error

    multiline = _coerce_bool(value=args.get("multiline"))

    if not pattern:
        return "Error: pattern is required"

    cwd = env.get("cwd", ".")
    if search_path:
        target = Path(search_path)
        if not target.is_absolute():
            target = Path(cwd) / target
        search_path = str(target)
    else:
        search_path = cwd

    target_path = Path(search_path)
    if not target_path.exists():
        return f"Error: Path does not exist: {target_path}"

    cmd = ["rg"]

    if output_mode == "files_with_matches" or output_mode is None:
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")

    if case_insensitive:
        cmd.append("-i")

    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])

    if show_line_numbers and output_mode == "content":
        cmd.append("-n")

    if before_ctx is not None and output_mode == "content":
        cmd.extend(["-B", str(before_ctx)])
    if after_ctx is not None and output_mode == "content":
        cmd.extend(["-A", str(after_ctx)])
    if context is not None and output_mode == "content":
        cmd.extend(["-C", str(context)])

    if glob_pattern:
        cmd.extend(["--glob", glob_pattern])

    if file_type:
        cmd.extend(["--type", file_type])

    cmd.append("--")
    cmd.append(pattern)
    cmd.append(search_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return (
            "Error: ripgrep (rg) is not installed. "
            "Install it with: brew install ripgrep (macOS) or apt install ripgrep (Ubuntu)"
        )
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"

    output = result.stdout
    if not output and result.returncode == 1:
        return "No matches found."
    if result.returncode not in (0, 1):
        return f"Error: rg failed with exit code {result.returncode}: {result.stderr}"

    lines = output.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    if offset is not None:
        lines = lines[offset:]
    if head_limit is not None:
        lines = lines[:head_limit]

    output = "\n".join(lines)

    if len(output.encode()) > MAX_SEARCH_RESULT_BYTES:
        output = output[:MAX_SEARCH_RESULT_BYTES].rsplit("\n", 1)[0]
        output += "\n\n[Results truncated]"

    return output if output else "No matches found."
