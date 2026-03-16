#! /usr/bin/env python3
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import base64
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any
import tempfile

RUNS_CONFIG_NAMES = ("runs.json", "runs.yaml")


@dataclass(frozen=True)
class ResolvedModelConfig:
    key: str
    model: str
    api_base: str
    api_key: str
    temperature: float | None
    context_length: int | None
    extra_params: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedAgentConfig:
    verbosity: int
    max_turns: int
    max_wait_seconds: float
    final_message: bool


def discover_repo_root(*, start_path: Path) -> Path:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        config_dir = candidate / "agent_evolve_v3"
        if any((config_dir / name).exists() for name in RUNS_CONFIG_NAMES):
            return candidate
    raise FileNotFoundError(
        f"Unable to locate repo root from {start_path} via agent_evolve_v3/runs.json."
    )


def load_workspace_agent(*, workspace_root: Path) -> object:
    agent_dir = workspace_root / "agent"
    module_path = agent_dir / "agent.py"
    if not module_path.exists():
        raise RuntimeError(f"Workspace agent not found at {module_path}")
    with workspace_imports(agent_dir=agent_dir):
        module = _load_module(
            module_name=f"workspace_agent_{abs(hash(module_path.resolve()))}",
            module_path=module_path,
        )
        agent_cls = getattr(module, "WorkspaceAgent", None)
        if agent_cls is None:
            raise RuntimeError("agent/agent.py must define WorkspaceAgent.")
        return agent_cls()


def build_runtime_config(
    *,
    workspace_root: Path,
    repo_root: Path,
    model_key: str,
    final_message_enabled: bool,
) -> object:
    model_cfg = resolve_model_config(repo_root=repo_root, model_key=model_key)
    agent_cfg = resolve_agent_config(
        repo_root=repo_root,
        final_message_enabled=final_message_enabled,
    )
    runtime_types = _load_runtime_types_module(workspace_root=workspace_root)
    return runtime_types.WorkspaceRuntimeConfig(
        model=runtime_types.WorkspaceModelConfig(
            model=model_cfg.model,
            api_base=model_cfg.api_base,
            api_key=model_cfg.api_key,
            temperature=model_cfg.temperature,
            context_length=model_cfg.context_length,
            extra_params=model_cfg.extra_params,
        ),
        agent_config={
            "verbosity": agent_cfg.verbosity,
            "max_turns": agent_cfg.max_turns,
            "max_wait_seconds": agent_cfg.max_wait_seconds,
            "final_message": agent_cfg.final_message,
        },
    )


def smoke_test_workspace(
    *,
    workspace_root: Path,
    repo_root: Path,
    model_key: str,
) -> dict[str, object]:
    agent = load_workspace_agent(workspace_root=workspace_root)
    build_runtime_config(
        workspace_root=workspace_root,
        repo_root=repo_root,
        model_key=model_key,
        final_message_enabled=True,
    )
    return {
        "workspace": str(workspace_root),
        "model_key": model_key,
        "agent_class": agent.__class__.__name__,
        "ready": hasattr(agent, "run_task"),
    }


def resolve_model_config(
    *,
    repo_root: Path,
    model_key: str,
) -> ResolvedModelConfig:
    catalog = json.loads((repo_root / "config.json").read_text(encoding="utf-8"))
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
    return ResolvedModelConfig(
        key=model_key,
        model=str(model_payload["model"]),
        api_base=str(model_payload["api_base"]),
        api_key=api_key,
        temperature=_maybe_float(value=model_payload.get("temperature")),
        context_length=_maybe_int(value=model_payload.get("context_length")),
        extra_params=_maybe_dict(value=model_payload.get("extra_params")),
    )


def resolve_agent_config(
    *,
    repo_root: Path,
    final_message_enabled: bool,
) -> ResolvedAgentConfig:
    catalog = json.loads((repo_root / "config.json").read_text(encoding="utf-8"))
    return ResolvedAgentConfig(
        verbosity=int(catalog.get("verbosity", 0)),
        max_turns=int(catalog.get("max_turns", 250)),
        max_wait_seconds=float(catalog.get("max_wait_seconds", 120.0)),
        final_message=final_message_enabled,
    )


def build_runtime_env_payload(
    *,
    repo_root: Path,
    model_key: str,
    final_message_enabled: bool,
) -> dict[str, str]:
    model_cfg = resolve_model_config(repo_root=repo_root, model_key=model_key)
    agent_cfg = resolve_agent_config(
        repo_root=repo_root,
        final_message_enabled=final_message_enabled,
    )
    env = {
        "WORKSPACE_CFG_MODEL": model_cfg.model,
        "WORKSPACE_CFG_API_BASE": model_cfg.api_base,
        "WORKSPACE_CFG_API_KEY": model_cfg.api_key,
        "WORKSPACE_CFG_VERBOSITY": str(agent_cfg.verbosity),
        "WORKSPACE_CFG_MAX_TURNS": str(agent_cfg.max_turns),
        "WORKSPACE_CFG_MAX_WAIT_SECONDS": str(agent_cfg.max_wait_seconds),
        "WORKSPACE_CFG_FINAL_MESSAGE": "1" if agent_cfg.final_message else "0",
    }
    if model_cfg.extra_params is not None:
        extra_params_json = json.dumps(model_cfg.extra_params, ensure_ascii=True)
        env["WORKSPACE_CFG_EXTRA_PARAMS_B64"] = base64.b64encode(
            extra_params_json.encode("utf-8")
        ).decode("ascii")
    if model_cfg.temperature is not None:
        env["WORKSPACE_CFG_TEMPERATURE"] = str(model_cfg.temperature)
    if model_cfg.context_length is not None:
        env["WORKSPACE_CFG_CONTEXT_LENGTH"] = str(model_cfg.context_length)
    return env


def stage_remote_bundle(
    *,
    workspace_root: Path,
    repo_root: Path,
) -> Path:
    agent_dir = workspace_root / "agent"
    if not agent_dir.exists():
        raise FileNotFoundError(f"Workspace agent directory not found: {agent_dir}")
    staging = Path(tempfile.mkdtemp(prefix="agent-evolve-v3-bundle-"))
    shutil.copytree(src=agent_dir, dst=staging / "agent", symlinks=True)
    shutil.copy2(
        src=repo_root / "agent_evolve_v3" / "services" / "remote_runner.py",
        dst=staging / "remote_runner.py",
    )
    return staging


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


@contextmanager
def workspace_imports(*, agent_dir: Path) -> Iterator[None]:
    module_prefixes = _workspace_module_prefixes(agent_dir=agent_dir)
    for prefix in module_prefixes:
        _purge_module_prefix(prefix=prefix)
    sys.path.insert(0, str(agent_dir))
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(agent_dir):
            sys.path.pop(0)


def _load_runtime_types_module(*, workspace_root: Path) -> ModuleType:
    agent_dir = workspace_root / "agent"
    module_path = agent_dir / "runtime_types.py"
    if not module_path.exists():
        raise RuntimeError(f"Workspace runtime types not found at {module_path}")
    with workspace_imports(agent_dir=agent_dir):
        return _load_module(module_name="runtime_types", module_path=module_path)


def _load_module(*, module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name=module_name,
        location=module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _workspace_module_prefixes(*, agent_dir: Path) -> list[str]:
    prefixes = ["runtime_types"]
    for entry in agent_dir.iterdir():
        if entry.name == "__pycache__":
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            prefixes.append(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            prefixes.append(entry.stem)
    ordered = []
    seen = set()
    for prefix in prefixes:
        if prefix not in seen:
            ordered.append(prefix)
            seen.add(prefix)
    return ordered


def _purge_module_prefix(*, prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


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
