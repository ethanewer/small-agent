from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

MAX_READ_SIZE = 2000
MAX_LINE_LENGTH = 2000
MAX_IMAGE_SIZE = 262144


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


def _detect_mime(path: Path, content: bytes) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return "text/plain"


def _is_visual(mime: str) -> bool:
    return mime.startswith("image/") or mime == "application/pdf"


def _truncate_line(line: str, max_len: int = MAX_LINE_LENGTH) -> str:
    if len(line) > max_len:
        return line[:max_len] + f"... [truncated, line exceeds {max_len} chars]"
    return line


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    raw_file_path = args.get("file_path") or args.get("path", "")
    file_path = str(raw_file_path).strip() if raw_file_path is not None else ""
    start_line, start_line_error = _coerce_optional_int(
        value=args.get("start_line"),
        name="start_line",
    )
    if start_line_error:
        return start_line_error

    end_line, end_line_error = _coerce_optional_int(
        value=args.get("end_line"),
        name="end_line",
    )
    if end_line_error:
        return end_line_error

    show_line_numbers = _coerce_bool(value=args.get("show_line_numbers", True))

    if not file_path:
        return "Error: file_path is required"

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(env.get("cwd", ".")) / path

    if not path.exists():
        return f"Error: File not found: {path}"

    if path.is_dir():
        return f"Error: {path} is a directory, not a file. Use shell with ls to list directories."

    try:
        raw = path.read_bytes()
    except PermissionError:
        return f"Error: Permission denied reading {path}"
    except Exception as e:
        return f"Error: Failed to read file {path}: {e}"

    mime = _detect_mime(path, raw)

    if _is_visual(mime):
        size = len(raw)
        if size > MAX_IMAGE_SIZE:
            return (
                f"Error: File size ({size} bytes) exceeds the maximum allowed size "
                f"of {MAX_IMAGE_SIZE} bytes"
            )
        import base64

        b64 = base64.b64encode(raw).decode()
        return f"[{mime} image, {size} bytes, base64-encoded]\ndata:{mime};base64,{b64}"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: Failed to read file as UTF-8 from {path}"

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    total_lines = len(lines)

    max_read = _coerce_optional_int(
        value=env.get("max_read_size"), name="max_read_size"
    )[0]
    if max_read is None or max_read <= 0:
        max_read = MAX_READ_SIZE

    if start_line is None and end_line is None:
        sl = 1
        el = min(total_lines, max_read)
    else:
        sl = max(1, start_line or 1)
        el = min(total_lines, end_line or total_lines)

    if total_lines == 0:
        return f"[File is empty: {path}]"

    sl = max(1, min(sl, total_lines))
    el = max(sl, min(el, total_lines))

    selected = lines[sl - 1 : el]

    output_lines = []
    for i, line in enumerate(selected, start=sl):
        truncated = _truncate_line(line)
        if show_line_numbers:
            output_lines.append(f"{i}:{truncated}")
        else:
            output_lines.append(truncated)

    result = "\n".join(output_lines)

    if el < total_lines:
        result += f"\n\n[Showing lines {sl}-{el} of {total_lines} total]"

    return result
