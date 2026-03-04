from __future__ import annotations

from typing import Any, Protocol


class ModelClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str: ...

    async def acomplete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str: ...
