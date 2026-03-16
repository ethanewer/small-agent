# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportImplicitRelativeImport=false

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

try:
    from agent_evolve_v3.workspace_support.workspace_agent_types import (
        WorkspaceModelConfig,
        WorkspaceRuntimeConfig,
    )
except ModuleNotFoundError:
    from workspace_agent_types import WorkspaceModelConfig, WorkspaceRuntimeConfig


@dataclass(frozen=True)
class WorkspaceMetadata:
    baseline: str
    created_from: str
    workspace_role: str | None = None


def load_workspace_metadata(*, workspace_root: Path) -> WorkspaceMetadata:
    payload = json.loads(
        (workspace_root / "workspace_metadata.json").read_text(encoding="utf-8")
    )
    return WorkspaceMetadata(
        baseline=str(payload["baseline"]),
        created_from=str(payload["created_from"]),
        workspace_role=_optional_text(value=payload.get("workspace_role")),
    )


def build_runtime_config(
    *,
    workspace_root: Path,
    model_key: str,
    final_message_enabled: bool,
) -> WorkspaceRuntimeConfig:
    catalog = json.loads(
        (workspace_root / "model_catalog.json").read_text(encoding="utf-8")
    )
    models = _as_dict(value=catalog.get("models"))
    if model_key not in models:
        supported = ", ".join(sorted(models))
        raise ValueError(
            f"Unknown model key '{model_key}'. Available keys: {supported}"
        )

    model_payload = _as_dict(value=models[model_key])
    api_key = resolve_api_key(config_api_key=model_payload.get("api_key"))
    if not api_key:
        raise ValueError(f"Unable to resolve API key for model '{model_key}'.")

    agent_config = {
        "verbosity": int(catalog.get("verbosity", 0)),
        "max_turns": int(catalog.get("max_turns", 250)),
        "max_wait_seconds": float(catalog.get("max_wait_seconds", 120.0)),
        "final_message": final_message_enabled,
    }
    return WorkspaceRuntimeConfig(
        model=WorkspaceModelConfig(
            model=str(model_payload["model"]),
            api_base=str(model_payload["api_base"]),
            api_key=api_key,
            temperature=_maybe_float(value=model_payload.get("temperature")),
            context_length=_maybe_int(value=model_payload.get("context_length")),
            extra_params=_maybe_dict(value=model_payload.get("extra_params")),
        ),
        agent_config=agent_config,
    )


def resolve_api_key(*, config_api_key: object) -> str | None:
    if not isinstance(config_api_key, str):
        return None
    raw = config_api_key.strip()
    if not raw:
        return None
    if raw.startswith("$"):
        raw = raw[1:]
    if raw.isupper() and raw.replace("_", "").isalnum():
        return resolve_env_value(env_name=raw)
    return raw


def resolve_env_value(*, env_name: str) -> str | None:
    current = os.environ.get(env_name)
    if current:
        return current
    if not shutil.which("zsh"):
        return None

    command = f'source ~/.zshrc >/dev/null 2>&1; printf %s "${{{env_name}}}"'
    completed = subprocess.run(
        ["zsh", "-ic", command],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def _as_dict(*, value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_int(*, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return None


def _maybe_float(*, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def _maybe_dict(*, value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _optional_text(*, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
