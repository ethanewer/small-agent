from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, overload


@dataclass
class Config:
    model: str
    api_base: str
    api_key: str
    temperature: float | None = None
    context_length: int | None = None
    extra_params: dict[str, Any] | None = None
    max_turns: int = 50
    max_wait_seconds: float = 60.0


@dataclass
class RunResult:
    exit_code: int
    success: bool
    history: list[dict[str, str]] | None = None


class ReasoningPayload(TypedDict):
    analysis: str
    plan: str


class CommandOutputPayload(TypedDict):
    keystrokes: str
    duration: float
    output: str


class IssuePayload(TypedDict):
    kind: str
    message: str


class DonePayload(TypedDict):
    message: str


class StoppedPayload(TypedDict):
    max_turns: int


class CompactionPayload(TypedDict):
    kind: str


class Logger(Protocol):
    @overload
    def log(
        self,
        *,
        event_type: Literal["reasoning"],
        payload: ReasoningPayload,
        turn: int,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: Literal["command_output"],
        payload: CommandOutputPayload,
        turn: int | None = ...,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: Literal["issue"],
        payload: IssuePayload,
        turn: int | None = ...,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: Literal["done"],
        payload: DonePayload,
        turn: int | None = ...,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: Literal["stopped"],
        payload: StoppedPayload,
        turn: int | None = ...,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: Literal["compaction"],
        payload: CompactionPayload,
        turn: int | None = ...,
    ) -> None: ...

    @overload
    def log(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        turn: int | None = ...,
    ) -> None: ...

    def log(  # pyright: ignore[reportInconsistentOverload]
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        turn: int | None = None,
    ) -> None: ...


class AgentRunner(Protocol):
    def __call__(
        self,
        *,
        instruction: str,
        config: Config,
        logger: Logger | None = None,
    ) -> RunResult: ...
