# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import time
from typing import Any, Callable, TypeVar

from context import Context, ToolCall

T = TypeVar("T")
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_LLM_ERROR_TYPE_NAMES = {
    "apiconnectionerror",
    "apitimeouterror",
    "ratelimiterror",
    "internalservererror",
}
TRANSIENT_LLM_ERROR_MESSAGE_SNIPPETS = (
    "connection error",
    "connection reset",
    "connection aborted",
    "connection timed out",
    "timed out",
    "temporary failure",
    "temporarily unavailable",
    "server disconnected",
    "rate limit",
    "overloaded",
)
LLM_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0)


@dataclass
class ChatResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model' or just 'model' into (provider_hint, model_id)."""
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider.lower(), model_id
    return "", model


def _resolve_openai_model(*, model: str, base_url: str | None) -> str:
    provider_hint, model_id = _parse_model_string(model)
    if "/" not in model:
        return model

    normalized_base = (base_url or "").lower()
    if "openrouter.ai" in normalized_base:
        # OpenRouter expects provider-prefixed model IDs like qwen/qwen3-coder-next.
        return model

    if provider_hint in {"openai", "anthropic"}:
        # OpenAI-compatible providers generally expect the bare model name.
        return model_id

    # Preserve provider-prefixed names for other OpenAI-compatible gateways.
    return model


def detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


def chat(
    context: Context,
    model: str,
    tools: list[dict],
    stream_callback: Any | None = None,
) -> ChatResponse:
    provider_hint, model_id = _parse_model_string(model)

    if not provider_hint:
        provider_hint = detect_provider()

    if provider_hint == "anthropic":
        return _chat_anthropic(context, model_id, tools, stream_callback)
    else:
        return _chat_openai(context, model, tools, stream_callback)


def _extract_status_code(*, error: Exception) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def _is_transient_llm_error(*, error: Exception) -> bool:
    if type(error).__name__.lower() in TRANSIENT_LLM_ERROR_TYPE_NAMES:
        return True
    if _extract_status_code(error=error) in TRANSIENT_HTTP_STATUS_CODES:
        return True
    message = str(error).lower()
    return any(snippet in message for snippet in TRANSIENT_LLM_ERROR_MESSAGE_SNIPPETS)


def _retry_delay_seconds(*, attempt: int) -> float:
    index = min(max(0, attempt - 1), len(LLM_RETRY_DELAYS_SECONDS) - 1)
    return LLM_RETRY_DELAYS_SECONDS[index]


def _call_with_retries(
    *,
    operation: Callable[[], T],
) -> T:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if not _is_transient_llm_error(error=error):
                raise
            attempt += 1
            time.sleep(_retry_delay_seconds(attempt=attempt))


def _chat_anthropic(
    context: Context,
    model_id: str,
    tools: list[dict],
    stream_callback: Any | None = None,
) -> ChatResponse:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
        )

    client = anthropic.Anthropic(api_key=api_key)

    system_text, messages = context.to_anthropic_messages()

    anthropic_tools = []
    for t in tools:
        func = t["function"]
        anthropic_tools.append(
            {
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"],
            }
        )

    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": context.max_tokens or 20480,
        "messages": messages,
    }
    if system_text:
        kwargs["system"] = system_text
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
    if stream_callback:
        return _stream_anthropic(client, kwargs, stream_callback)

    response = _call_with_retries(
        operation=lambda: client.messages.create(**kwargs),
    )

    content_text = ""
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )

    finish_reason = (
        "stop" if response.stop_reason == "end_turn" else response.stop_reason
    )
    if response.stop_reason == "tool_use":
        finish_reason = "tool_calls"

    usage = None
    if response.usage:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    return ChatResponse(
        content=content_text or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _stream_anthropic(
    client: Any,
    kwargs: dict[str, Any],
    stream_callback: Any,
) -> ChatResponse:
    attempt = 0
    while True:
        content_text = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None
        finish_reason = None
        usage = None

        try:
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            block = event.content_block
                            if hasattr(block, "type") and block.type == "tool_use":
                                current_tool = {
                                    "id": block.id,
                                    "name": block.name,
                                    "json_str": "",
                                }
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, "type"):
                                if delta.type == "text_delta":
                                    content_text += delta.text
                                    stream_callback(delta.text)
                                elif delta.type == "input_json_delta" and current_tool:
                                    current_tool["json_str"] += delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool:
                                try:
                                    args = (
                                        json.loads(current_tool["json_str"])
                                        if current_tool["json_str"]
                                        else {}
                                    )
                                except json.JSONDecodeError:
                                    args = {}
                                tool_calls.append(
                                    ToolCall(
                                        id=current_tool["id"],
                                        name=current_tool["name"],
                                        arguments=args,
                                    )
                                )
                                current_tool = None
                        elif event.type == "message_delta":
                            if hasattr(event, "delta") and hasattr(
                                event.delta, "stop_reason"
                            ):
                                sr = event.delta.stop_reason
                                if sr == "end_turn":
                                    finish_reason = "stop"
                                elif sr == "tool_use":
                                    finish_reason = "tool_calls"
                                else:
                                    finish_reason = sr
                            if hasattr(event, "usage") and event.usage:
                                usage = {"output_tokens": event.usage.output_tokens}

                msg = stream.get_final_message()
                if msg and msg.usage:
                    usage = {
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens,
                    }
            return ChatResponse(
                content=content_text or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as error:
            has_partial_response = bool(
                content_text or tool_calls or current_tool is not None
            )
            if has_partial_response or not _is_transient_llm_error(error=error):
                raise
            attempt += 1
            time.sleep(_retry_delay_seconds(attempt=attempt))


def _chat_openai(
    context: Context,
    model: str,
    tools: list[dict],
    stream_callback: Any | None = None,
) -> ChatResponse:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_URL") or os.environ.get("OPENAI_BASE_URL")

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    messages = context.to_api_messages()

    model_id = _resolve_openai_model(model=model, base_url=base_url)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if context.max_tokens:
        kwargs["max_tokens"] = context.max_tokens
    if tools:
        kwargs["tools"] = tools
    if context.extra_params:
        kwargs["extra_body"] = context.extra_params

    if stream_callback:
        return _stream_openai(client, kwargs, stream_callback)

    response = _call_with_retries(
        operation=lambda: client.chat.completions.create(**kwargs),
    )

    choice = response.choices[0]
    msg = choice.message

    content_text = msg.content
    tool_calls = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = (
                    json.loads(tc.function.arguments) if tc.function.arguments else {}
                )
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    finish_reason = choice.finish_reason
    if finish_reason == "tool_calls":
        pass
    elif finish_reason == "stop" and not tool_calls:
        pass
    elif tool_calls:
        finish_reason = "tool_calls"

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return ChatResponse(
        content=content_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _stream_openai(
    client: Any,
    kwargs: dict[str, Any],
    stream_callback: Any,
) -> ChatResponse:
    kwargs["stream"] = True
    attempt = 0
    while True:
        content_text = ""
        tool_calls_map: dict[int, dict[str, str]] = {}
        finish_reason = None

        try:
            response = client.chat.completions.create(**kwargs)

            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                fr = chunk.choices[0].finish_reason

                if delta.content:
                    content_text += delta.content
                    stream_callback(delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name
                                if tc_delta.function and tc_delta.function.name
                                else "",
                                "arguments_str": "",
                            }
                        entry = tool_calls_map[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments_str"] += tc_delta.function.arguments

                if fr:
                    finish_reason = fr

            tool_calls = []
            for idx in sorted(tool_calls_map.keys()):
                entry = tool_calls_map[idx]
                try:
                    args = (
                        json.loads(entry["arguments_str"])
                        if entry["arguments_str"]
                        else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=entry["id"],
                        name=entry["name"],
                        arguments=args,
                    )
                )

            if tool_calls and finish_reason != "tool_calls":
                finish_reason = "tool_calls"

            return ChatResponse(
                content=content_text or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=None,
            )
        except Exception as error:
            if (
                content_text
                or tool_calls_map
                or not _is_transient_llm_error(error=error)
            ):
                raise
            attempt += 1
            time.sleep(_retry_delay_seconds(attempt=attempt))
