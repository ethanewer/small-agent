from __future__ import annotations

from typing import Protocol


class Environment(Protocol):
    def run_command(
        self, *, command: str, timeout_seconds: float | None = None
    ) -> str: ...

    def wait(self, *, duration_seconds: float) -> str: ...
