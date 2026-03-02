from __future__ import annotations

from typing import Literal

ProviderKind = Literal[
    "openrouter",
    "openai",
    "openai_compatible_local",
    "other_openai_compatible",
]


def detect_provider_kind(*, api_base: str) -> ProviderKind:
    normalized_base = api_base.rstrip("/").lower()
    if normalized_base.endswith("openrouter.ai/api/v1"):
        return "openrouter"

    if normalized_base.endswith("api.openai.com/v1"):
        return "openai"

    if "localhost" in normalized_base or "127.0.0.1" in normalized_base:
        return "openai_compatible_local"

    return "other_openai_compatible"


def normalize_openai_compatible_model(*, model: str, api_base: str) -> str:
    provider_kind = detect_provider_kind(api_base=api_base)
    normalized = model.strip()
    if normalized.startswith("openai/"):
        normalized = normalized.removeprefix("openai/")

    if provider_kind == "openrouter" and normalized.startswith("openrouter/"):
        normalized = normalized.removeprefix("openrouter/")

    if provider_kind != "openrouter" and normalized.startswith("openrouter/"):
        normalized = normalized.removeprefix("openrouter/")

    return normalized


def opencode_model_arg(*, model: str, api_base: str) -> str:
    provider_kind = detect_provider_kind(api_base=api_base)
    normalized_model = normalize_openai_compatible_model(model=model, api_base=api_base)
    if provider_kind == "openrouter":
        return f"openrouter/{normalized_model}"

    return f"openai/{normalized_model}"


def preflight_agent_model_compatibility(
    *,
    agent_key: str,
    model: str,
    api_base: str,
) -> str | None:
    provider_kind = detect_provider_kind(api_base=api_base)
    normalized_model = normalize_openai_compatible_model(model=model, api_base=api_base)

    if agent_key == "qwen" and provider_kind == "openai":
        lowered = normalized_model.lower()
        if "codex" in lowered:
            return (
                "qwen agent is incompatible with OpenAI Codex-style models on "
                "chat-completions endpoints. Choose a chat-completions-capable model "
                "or use terminus-2 for this model."
            )

    if agent_key == "claude":
        if "claude" not in normalized_model.lower():
            return (
                "claude agent only supports Claude-family model IDs. "
                "Choose a Claude model (for example claude-sonnet-4-6) "
                "or use qwen/opencode/terminus-2 for non-Claude models."
            )

    return None
