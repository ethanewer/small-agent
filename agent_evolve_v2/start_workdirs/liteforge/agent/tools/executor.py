# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools import (
    fetch,
    fs_patch,
    fs_read,
    fs_remove,
    fs_search,
    fs_undo,
    fs_write,
    shell,
)
from tools.todo import TodoManager


class ToolExecutor:
    def __init__(self, env: dict[str, Any]) -> None:
        self.env = env
        self.snapshots: dict[str, str | None] = {}
        self.files_accessed: set[str] = set()
        self.todo_manager = TodoManager()

    def _normalize_path(self, raw_path: str) -> str:
        p = Path(raw_path)
        if p.is_absolute():
            return str(p)
        return str(Path(self.env.get("cwd", ".")) / p)

    def _require_prior_read(self, raw_path: str, action: str) -> str | None:
        target = self._normalize_path(raw_path)
        if target in self.files_accessed or raw_path in self.files_accessed:
            return None
        return (
            f"You must read the file with the read tool before attempting to {action}."
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool call. Returns (output_text, is_error)."""
        try:
            return self._dispatch(tool_name, arguments), False
        except Exception as e:
            return f"Error executing {tool_name}: {e}", True

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "read":
            result = fs_read.execute(args, self.env)
            file_path = args.get("file_path") or args.get("path", "")
            if file_path:
                self.files_accessed.add(self._normalize_path(file_path))
            return result

        if tool_name == "write":
            file_path = args.get("file_path") or args.get("path", "")
            if args.get("overwrite", False) and file_path:
                err = self._require_prior_read(file_path, "overwrite it")
                if err:
                    return f"Error: {err}"
            return fs_write.execute(args, self.env, self.snapshots)

        if tool_name == "patch":
            file_path = args.get("file_path") or args.get("path", "")
            if file_path:
                err = self._require_prior_read(file_path, "edit it")
                if err:
                    return f"Error: {err}"
            return fs_patch.execute(args, self.env, self.snapshots)

        if tool_name == "fs_search":
            return fs_search.execute(args, self.env)

        if tool_name == "shell":
            return shell.execute(args, self.env)

        if tool_name == "fetch":
            return fetch.execute(args, self.env)

        if tool_name == "remove":
            return fs_remove.execute(args, self.env, self.snapshots)

        if tool_name == "undo":
            return fs_undo.execute(args, self.env, self.snapshots)

        if tool_name == "todo_write":
            from tools.todo import execute_write

            return execute_write(args, self.todo_manager)

        if tool_name == "todo_read":
            from tools.todo import execute_read

            return execute_read(self.todo_manager)

        return f"Error: Unknown tool: {tool_name}"
