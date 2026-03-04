from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.registry import get_agent
from benchmark.adapters import AgentBenchmarkAdapter
from benchmark.runtime_config import build_runtime_cfg
from benchmark.terminalbench_tb_adapter import TerminalBenchTBAdapter
import cli as cli_module
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless benchmark batches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def _load_rows(*, input_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(
        input_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Invalid JSON on line {line_index} in {input_jsonl}: {err}"
            ) from err
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Invalid row on line {line_index} in {input_jsonl}: expected JSON object."
            )

        rows.append(parsed)

    return rows


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = cli_module.load_config(args.config)
    runtime_cfg = build_runtime_cfg(
        cfg=cfg,
        agent_key=args.agent,
        model_key=args.model,
    )
    adapter = AgentBenchmarkAdapter(
        agent=get_agent(args.agent),
        runtime_cfg=runtime_cfg,
        console=console,
    )
    tb_adapter = TerminalBenchTBAdapter(adapter=adapter)

    rows = _load_rows(input_jsonl=args.input_jsonl)
    tb_adapter.run_samples_to_jsonl(samples=rows, output_path=args.output_jsonl)


if __name__ == "__main__":
    main()
