from __future__ import annotations

from pathlib import Path
from typing import Any


def execute(
    args: dict[str, Any],
    env: dict[str, Any],
    snapshots: dict[str, str | None],
) -> str:
    file_path = args.get("file_path") or args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = args.get("replace_all", False)

    if not file_path:
        return "Error: file_path is required"

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(env.get("cwd", ".")) / path

    if not path.exists():
        return f"Error: File not found: {path}"

    try:
        content = path.read_text()
    except Exception as e:
        return f"Error: Failed to read file {path}: {e}"

    if not old_string:
        return "Error: old_string cannot be empty"

    if old_string == new_string:
        return "Error: old_string and new_string must be different"

    count = content.count(old_string)

    if count == 0:
        return (
            f"Error: old_string not found in {path}. "
            "Make sure the text matches exactly, including whitespace and indentation."
        )

    if count > 1 and not replace_all:
        return (
            f"Error: old_string found {count} times in {path}. "
            "Provide more context to make it unique, or use replace_all=true."
        )

    snapshots[str(path)] = content

    if replace_all:
        new_content = content.replace(old_string, new_string)
        path.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {path}"
    else:
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content)
        return f"Replaced 1 occurrence in {path}"
