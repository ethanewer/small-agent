# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportImplicitRelativeImport=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from benchmark_cache import resolve_latest_visible_run_dir
from critic_tools import summarize_job, write_critic_outputs


def main(argv: list[str]) -> int:
    args = parse_args(argv=argv)
    workspace_root = Path(__file__).resolve().parent
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else resolve_latest_visible_run_dir(workspace_root=workspace_root)
    )
    benchmark_summary = json.loads(
        (artifacts_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    _summary, critic_summary = summarize_job(
        harbor_job_dir=Path(str(benchmark_summary["harbor_job_dir"])),
        max_failure_items=args.max_failures,
        max_success_items=args.max_successes,
    )
    write_critic_outputs(
        output_dir=artifacts_dir,
        benchmark_summary=_summary,
        critic_summary=critic_summary,
    )
    print(json.dumps(asdict(critic_summary), ensure_ascii=True))
    return 0


def parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a structured critic summary for benchmark artifacts.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--max-successes", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
