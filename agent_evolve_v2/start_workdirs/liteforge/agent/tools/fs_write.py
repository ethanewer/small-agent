# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

import os
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
        return (
            "Error: Cannot overwrite existing file: overwrite flag not set. "
            f"File already exists at {path}"
        )

    snapshots[str(path)] = path.read_text(errors="replace") if path.exists() else None

    path.parent.mkdir(parents=True, exist_ok=True)

    target_line_ending = "\n"
    if path.exists() and overwrite:
        existing = path.read_text(errors="replace")
        target_line_ending = "\r\n" if "\r\n" in existing else "\n"
    elif os.name == "nt":
        target_line_ending = "\r\n"

    normalized_content = content.replace("\r\n", "\n").replace("\n", target_line_ending)

    path.write_text(normalized_content)

    action = "Updated" if snapshots.get(str(path)) is not None else "Created"
    return f"{action} file: {path}"
