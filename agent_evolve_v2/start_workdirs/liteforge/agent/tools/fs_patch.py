# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

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

    line_ending = "\r\n" if "\r\n" in content else "\n"
    normalized_old = old_string.replace("\r\n", "\n").replace("\n", line_ending)
    normalized_new = new_string.replace("\r\n", "\n").replace("\n", line_ending)

    if not old_string:
        return "Error: old_string cannot be empty"

    snapshots[str(path)] = content

    count = content.count(normalized_old)
    if count == 0:
        return (
            "Error: Could not find match for search text: "
            f"'{old_string}'. File may have changed externally, consider reading the file again."
        )

    if not replace_all and count > 1:
        return (
            "Error: Multiple matches found for search text: "
            f"'{old_string}'. Either provide a more specific search pattern "
            "or use replace_all to replace all occurrences."
        )

    if replace_all:
        new_content = content.replace(normalized_old, normalized_new)
        path.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {path}"

    new_content = content.replace(normalized_old, normalized_new, 1)
    path.write_text(new_content)
    return f"Replaced 1 occurrence in {path}"
