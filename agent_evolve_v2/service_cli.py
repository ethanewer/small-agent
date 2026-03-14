#! /usr/bin/env python3
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from agent_evolve_v2.benchmark import execute_workspace_benchmark
from agent_evolve_v2.critic import summarize_job, write_critic_outputs
from agent_evolve_v2.service_runtime import (
    discover_repo_root,
    smoke_test_workspace,
)
from agent_evolve_v2.workspace_support.benchmark_cache import (
    resolve_latest_visible_run_dir,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hidden workspace services for validation, benchmarking, and critique.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--workspace", type=Path, required=True)
    validate.add_argument("--model-key", type=str, required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--workspace", type=Path, required=True)
    benchmark.add_argument("--model-key", type=str, required=True)
    benchmark.add_argument("--request-label", type=str, default="manual")
    benchmark.add_argument("--result-json-out", type=Path, default=None)
    benchmark.add_argument(
        "--record-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    critique = subparsers.add_parser("critique")
    critique.add_argument("--workspace", type=Path, required=True)
    critique.add_argument("--artifacts-dir", type=Path, default=None)
    critique.add_argument("--max-failures", type=int, default=5)
    critique.add_argument("--max-successes", type=int, default=1)
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
        )
    if args.command == "critique":
        return _run_critique(args=args)
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


def _run_critique(*, args: argparse.Namespace) -> int:
    workspace_root = args.workspace.resolve()
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else resolve_latest_visible_run_dir(workspace_root=workspace_root)
    )
    benchmark_summary = json.loads(
        (artifacts_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    summary, critic_summary = summarize_job(
        harbor_job_dir=Path(str(benchmark_summary["harbor_job_dir"])),
        max_failure_items=int(args.max_failures),
        max_success_items=int(args.max_successes),
    )
    write_critic_outputs(
        output_dir=artifacts_dir,
        benchmark_summary=summary,
        critic_summary=critic_summary,
    )
    print(json.dumps(asdict(critic_summary), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
