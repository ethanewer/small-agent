# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

from pathlib import Path
from typing import Any


def execute(
    args: dict[str, Any],
    env: dict[str, Any],
    snapshots: dict[str, str | None],
) -> str:
    file_path = args.get("path", "")

    if not file_path:
        return "Error: path is required"

    path = Path(file_path)
    if not path.is_absolute():
        path = Path(env.get("cwd", ".")) / path

    if not path.exists():
        return f"Error: File not found: {path}"

    if path.is_dir():
        return (
            f"Error: {path} is a directory. Use shell with rm -r to remove directories."
        )

    try:
        content = path.read_text(errors="replace")
    except Exception:
        content = ""

    snapshots[str(path)] = content

    try:
        path.unlink()
    except Exception as e:
        return f"Error: Failed to remove {path}: {e}"

    return f"Removed file: {path}"
