from __future__ import annotations

from typing import Any


class TodoManager:
    def __init__(self) -> None:
        self._todos: list[dict[str, str]] = []

    def update_todos(self, todos: list[dict[str, Any]]) -> None:
        for todo in todos:
            todo_id = todo.get("id", "")
            content = todo.get("content", "")
            status = todo.get("status", "pending")

            existing = next((t for t in self._todos if t["id"] == todo_id), None)
            if existing:
                if content:
                    existing["content"] = content
                existing["status"] = status
            else:
                self._todos.append(
                    {
                        "id": todo_id,
                        "content": content,
                        "status": status,
                    }
                )

    def format_todos(self) -> str:
        if not self._todos:
            return "No todos."

        lines = []
        for t in self._todos:
            status = t["status"]
            marker = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(
                status, "[ ]"
            )
            lines.append(f"{marker} {t['id']}: {t['content']}")
        return "\n".join(lines)


def execute_write(args: dict[str, Any], manager: TodoManager) -> str:
    todos = args.get("todos", [])
    if not todos:
        return "Error: todos array is required"

    manager.update_todos(todos)
    return manager.format_todos()


def execute_read(manager: TodoManager) -> str:
    return manager.format_todos()
