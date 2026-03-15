# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

MAX_READ_SIZE = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 262144
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
    # Magic byte detection first, matching forge-repo behavior.
    if content.startswith(b"%PDF"):
        return "application/pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"RIFF") and b"WEBP" in content[:32]:
        return "image/webp"

    ext = path.suffix.lower()
    if ext in {
        ".txt",
        ".md",
        ".rs",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".js",
        ".ts",
        ".py",
        ".sh",
    }:
        return "text/plain"
    if ext == ".ipynb":
        return "application/json"

    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return "text/plain"


def _is_visual(mime: str) -> bool:
    return mime.startswith("image/") or mime == "application/pdf"


def _truncate_line(line: str, max_len: int = MAX_LINE_LENGTH) -> str:
    if len(line) > max_len:
        return "".join(ch for idx, ch in enumerate(line) if idx < max_len) + (
            f"... [truncated, line exceeds {max_len} chars]"
        )
    return line


def _resolve_range(
    *,
    start_line: int | None,
    end_line: int | None,
    max_size: int,
) -> tuple[int, int]:
    if max_size <= 0:
        return 1, 1
    s0 = max(start_line or 1, 1)
    e0 = end_line if end_line is not None else s0 + max_size - 1
    start = max(min(s0, e0), 1)
    end = max(s0, e0)
    end = min(end, start + max_size - 1)
    return start, end


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

    max_file_size = _coerce_optional_int(
        value=env.get("max_file_size"), name="max_file_size"
    )[0]
    if max_file_size is None or max_file_size <= 0:
        max_file_size = MAX_FILE_SIZE
    max_image_size = _coerce_optional_int(
        value=env.get("max_image_size"), name="max_image_size"
    )[0]
    if max_image_size is None or max_image_size <= 0:
        max_image_size = MAX_IMAGE_SIZE

    initial_size_limit = max(max_file_size, max_image_size)
    if len(raw) > initial_size_limit:
        return (
            f"Error: File size ({len(raw)} bytes) exceeds the maximum allowed size "
            f"of {initial_size_limit} bytes"
        )

    mime = _detect_mime(path, raw)

    if _is_visual(mime):
        size = len(raw)
        if size > max_image_size:
            return (
                f"Error: File size ({size} bytes) exceeds the maximum allowed size "
                f"of {max_image_size} bytes"
            )
        import base64

        b64 = base64.b64encode(raw).decode()
        return f"[{mime} image, {size} bytes, base64-encoded]\ndata:{mime};base64,{b64}"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: Failed to read file as UTF-8 from {path}"

    lines = text.splitlines()
    total_lines = len(lines)

    max_read = _coerce_optional_int(
        value=env.get("max_read_size"), name="max_read_size"
    )[0]
    if max_read is None or max_read <= 0:
        max_read = MAX_READ_SIZE

    if total_lines == 0:
        return ""

    resolved_start, resolved_end = _resolve_range(
        start_line=start_line,
        end_line=end_line,
        max_size=max_read,
    )

    start_pos = min(max(resolved_start - 1, 0), total_lines - 1)
    end_pos = min(max(resolved_end - 1, 0), total_lines - 1)

    selected = lines[start_pos : end_pos + 1]

    output_lines = []
    for i, line in enumerate(selected, start=start_pos + 1):
        truncated = _truncate_line(line)
        if show_line_numbers:
            output_lines.append(f"{i}:{truncated}")
        else:
            output_lines.append(truncated)

    result = "\n".join(output_lines)

    if end_pos + 1 < total_lines:
        result += (
            f"\n\n[Showing lines {start_pos + 1}-{end_pos + 1} of {total_lines} total]"
        )

    return result
