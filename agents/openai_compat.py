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


def preflight_agent_model_compatibility(
    *,
    agent_key: str,
    model: str,
    api_base: str,
) -> str | None:
    del agent_key, model, api_base
    return None
