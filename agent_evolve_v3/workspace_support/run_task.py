# pyright: reportImplicitRelativeImport=false, reportMissingTypeStubs=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from rich.console import Console

from workspace_config import build_runtime_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the copied workspace harness for a single instruction.",
    )
    parser.add_argument("--model-key", type=str, required=True)
    parser.add_argument("--no-final-message", action="store_true", default=False)
    parser.add_argument("instruction", type=str)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv=argv)
    workspace_root = Path(__file__).resolve().parent
    runtime_cfg = build_runtime_config(
        workspace_root=workspace_root,
        model_key=args.model_key,
        final_message_enabled=not args.no_final_message,
    )
    agent = _load_workspace_agent(workspace_root=workspace_root)
    result = agent.run_task(
        instruction=args.instruction,
        cfg=runtime_cfg,
        console=Console(),
        task_id="workspace-benchmark",
    )
    return result.exit_code


def _load_workspace_agent(*, workspace_root: Path):
    module_path = workspace_root / "agents" / "agent.py"
    spec = importlib.util.spec_from_file_location(
        name="workspace_agent_module",
        location=module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load workspace agent from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_cls = getattr(module, "WorkspaceAgent", None)
    if agent_cls is None:
        raise RuntimeError("agents/agent.py must define WorkspaceAgent.")
    return agent_cls()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
