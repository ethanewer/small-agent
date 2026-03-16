#! /usr/bin/env python3
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent_evolve_v3.benchmark import execute_workspace_benchmark
from agent_evolve_v3.service_runtime import (
    discover_repo_root,
    smoke_test_workspace,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hidden workspace services for validation and benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--workspace", type=Path, required=True)
    validate.add_argument("--model-key", type=str, required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--workspace", type=Path, required=True)
    benchmark.add_argument("--model-key", type=str, required=True)
    benchmark.add_argument("--request-label", type=str, default="manual")
    benchmark.add_argument("--artifacts-dir", type=Path, default=None)
    benchmark.add_argument(
        "--benchmark-preset",
        choices=("official", "smoke"),
        default="official",
    )
    benchmark.add_argument("--result-json-out", type=Path, default=None)
    benchmark.add_argument(
        "--record-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv=argv)
    if args.command == "validate":
        return _run_validate(args=args)
    if args.command == "benchmark":
        return execute_workspace_benchmark(
            workspace_path=args.workspace,
            model_key=args.model_key,
            result_json_out=args.result_json_out,
            record_visible=bool(args.record_visible),
            request_label=args.request_label,
            artifacts_dir=args.artifacts_dir,
            benchmark_preset=args.benchmark_preset,
        )
    raise RuntimeError(f"Unknown command: {args.command}")


def _run_validate(*, args: argparse.Namespace) -> int:
    workspace_root = args.workspace.resolve()
    repo_root = discover_repo_root(start_path=workspace_root)
    payload = smoke_test_workspace(
        workspace_root=workspace_root,
        repo_root=repo_root,
        model_key=args.model_key,
    )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
