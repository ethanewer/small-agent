#! /usr/bin/env python3
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportAttributeAccessIssue=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import argparse
import json
import importlib.util
import os
from pathlib import Path
import sys

from rich.console import Console


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the uploaded workspace agent inside Harbor.",
    )
    parser.add_argument("--no-final-message", action="store_true", default=False)
    parser.add_argument("instruction", type=str)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv=argv)
    workspace_root = Path(__file__).resolve().parent
    runtime_cfg = _build_runtime_config(workspace_root=workspace_root)
    agent = _load_workspace_agent(workspace_root=workspace_root)
    result = agent.run_task(
        instruction=args.instruction,
        cfg=runtime_cfg,
        console=Console(),
        task_id="workspace-benchmark",
    )
    return int(getattr(result, "exit_code", 1))


def _build_runtime_config(*, workspace_root: Path) -> object:
    agent_dir = workspace_root / "agent"
    runtime_types_path = agent_dir / "runtime_types.py"
    with _workspace_imports(agent_dir=agent_dir):
        spec = importlib.util.spec_from_file_location(
            name="runtime_types",
            location=runtime_types_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load runtime types from {runtime_types_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["runtime_types"] = module
        spec.loader.exec_module(module)
        extra_params = json.loads(
            os.environ.get("WORKSPACE_CFG_EXTRA_PARAMS_JSON", "null")
        )
        temperature = _maybe_float(value=os.environ.get("WORKSPACE_CFG_TEMPERATURE"))
        context_length = _maybe_int(
            value=os.environ.get("WORKSPACE_CFG_CONTEXT_LENGTH")
        )
        return module.WorkspaceRuntimeConfig(
            model=module.WorkspaceModelConfig(
                model=os.environ["WORKSPACE_CFG_MODEL"],
                api_base=os.environ["WORKSPACE_CFG_API_BASE"],
                api_key=os.environ["WORKSPACE_CFG_API_KEY"],
                temperature=temperature,
                context_length=context_length,
                extra_params=extra_params if isinstance(extra_params, dict) else None,
            ),
            agent_config={
                "verbosity": int(os.environ.get("WORKSPACE_CFG_VERBOSITY", "0")),
                "max_turns": int(os.environ.get("WORKSPACE_CFG_MAX_TURNS", "250")),
                "max_wait_seconds": float(
                    os.environ.get("WORKSPACE_CFG_MAX_WAIT_SECONDS", "120.0")
                ),
                "final_message": (
                    os.environ.get("WORKSPACE_CFG_FINAL_MESSAGE", "1") == "1"
                ),
            },
        )


def _load_workspace_agent(*, workspace_root: Path) -> object:
    agent_dir = workspace_root / "agent"
    agent_path = agent_dir / "agent.py"
    with _workspace_imports(agent_dir=agent_dir):
        spec = importlib.util.spec_from_file_location(
            name=f"workspace_agent_{abs(hash(agent_path.resolve()))}",
            location=agent_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load workspace agent from {agent_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        agent_cls = getattr(module, "WorkspaceAgent", None)
        if agent_cls is None:
            raise RuntimeError("agent/agent.py must define WorkspaceAgent.")
        return agent_cls()


class _workspace_imports:
    def __init__(self, *, agent_dir: Path) -> None:
        self.agent_dir = agent_dir

    def __enter__(self) -> None:
        for prefix in _workspace_module_prefixes(agent_dir=self.agent_dir):
            _purge_module_prefix(prefix=prefix)
        sys.path.insert(0, str(self.agent_dir))
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        if sys.path and sys.path[0] == str(self.agent_dir):
            sys.path.pop(0)


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


def _maybe_int(*, value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _maybe_float(*, value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
