from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None


@dataclass
class Context:
    """Holds the full conversation state sent to the LLM."""

    messages: list[Message] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    max_tokens: int | None = None
    extra_params: dict[str, Any] | None = None
    token_count: int = 0

    def set_system_messages(self, texts: list[str]) -> None:
        self.messages = [m for m in self.messages if m.role != "system"]
        system_msgs = [Message(role="system", content=t) for t in texts if t]
        self.messages = system_msgs + self.messages

    def add_user_message(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text))

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_tool_result(self, result: ToolResult) -> None:
        self.messages.append(Message(role="tool", tool_result=result))

    def append_turn(
        self,
        assistant_content: str | None,
        tool_calls: list[ToolCall] | None,
        tool_results: list[tuple[ToolCall, ToolResult]],
    ) -> None:
        self.add_assistant_message(assistant_content, tool_calls)
        for _call, result in tool_results:
            self.add_tool_result(result)

    def clone(self) -> Context:
        return copy.deepcopy(self)

    def to_api_messages(self) -> list[dict]:
        """Convert to the format expected by the OpenAI-compatible APIs.

        Consecutive system messages are merged into a single message so
        that providers which only accept one system message (e.g. Together)
        do not reject the request.
        """
        system_parts: list[str] = []
        api_msgs: list[dict] = []
        for msg in self.messages:
            if msg.role == "system":
                system_parts.append(msg.content or "")
                continue

            if system_parts:
                api_msgs.append(
                    {"role": "system", "content": "\n\n".join(system_parts)}
                )
                system_parts = []

            if msg.role == "user":
                api_msgs.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}
                if msg.tool_calls:
                    entry["content"] = msg.content or ""
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": _json_dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                else:
                    entry["content"] = msg.content or ""
                api_msgs.append(entry)
            elif msg.role == "tool" and msg.tool_result:
                api_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                )

        if system_parts:
            api_msgs.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

        return api_msgs

    def to_anthropic_messages(self) -> tuple[str, list[dict]]:
        """Convert to Anthropic format: (system_text, messages)."""
        system_parts: list[str] = []
        msgs: list[dict] = []

        for msg in self.messages:
            if msg.role == "system":
                system_parts.append(msg.content or "")
            elif msg.role == "user":
                msgs.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant":
                if msg.tool_calls:
                    content_blocks: list[dict] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    msgs.append({"role": "assistant", "content": content_blocks})
                else:
                    msgs.append({"role": "assistant", "content": msg.content or ""})
            elif msg.role == "tool" and msg.tool_result:
                tr = msg.tool_result
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                                **({"is_error": True} if tr.is_error else {}),
                            }
                        ],
                    }
                )

        return "\n\n".join(system_parts), msgs


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
