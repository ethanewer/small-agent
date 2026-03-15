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

    key = str(path)
    if key not in snapshots:
        return f"Error: No snapshot found for {path}. Cannot undo."

    previous = snapshots.pop(key)

    if previous is None:
        if path.exists():
            path.unlink()
        return f"Reverted {path} (file removed, was newly created)"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(previous)
        return f"Reverted {path} to previous state"
