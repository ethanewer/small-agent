from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.interface import AgentModelConfig, AgentRuntimeConfig
from agents.registry import get_agent
from benchmark.adapters import AgentBenchmarkAdapter
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


def _build_runtime_cfg(
    *, cfg: Any, agent_key: str, model_key: str
) -> AgentRuntimeConfig:
    model_cfg = cfg.models[model_key]
    resolved_api_key = cli_module.resolve_api_key(model_cfg.api_key)
    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for model '{model_key}'. Set env var or literal api_key."
        )

    agent_options = {
        "verbosity": cfg.verbosity,
        "max_turns": cfg.max_turns,
        "max_wait_seconds": cfg.max_wait_seconds,
        **dict(cfg.agents.get(agent_key, {})),
    }
    return AgentRuntimeConfig(
        agent_key=agent_key,
        model=AgentModelConfig(
            model=model_cfg.model,
            api_base=model_cfg.api_base,
            api_key=resolved_api_key,
            temperature=model_cfg.temperature,
        ),
        agent_config=agent_options,
    )


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = cli_module.load_config(args.config)
    runtime_cfg = _build_runtime_cfg(
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
