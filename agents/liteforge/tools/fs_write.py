from __future__ import annotations

from pathlib import Path
from typing import Any


def execute(
    args: dict[str, Any],
    env: dict[str, Any],
    snapshots: dict[str, str | None],
) -> str:
    file_path = args.get("file_path") or args.get("path", "")
    content = args.get("content", "")
    overwrite = args.get("overwrite", False)

    if not file_path:
        return "Error: file_path is required"

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(env.get("cwd", ".")) / path

    if path.exists() and not overwrite:
        existing = path.read_text(errors="replace")
        return (
            f"Error: File already exists at {path}. "
            f"Set overwrite=true to overwrite. Current content:\n{existing}"
        )

    snapshots[str(path)] = path.read_text(errors="replace") if path.exists() else None

    path.parent.mkdir(parents=True, exist_ok=True)

    if content and not content.endswith("\n"):
        content += "\n"

    path.write_text(content)

    action = "Updated" if snapshots.get(str(path)) is not None else "Created"
    return f"{action} file: {path}"
