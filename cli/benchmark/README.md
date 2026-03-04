# Benchmark adapters

This directory contains runtime adapters for benchmark-style execution.

## TerminalBench

- `terminalbench_tb_adapter.py`: maps TerminalBench-style samples into core `Task` and runs them.
- `terminalbench_harbor_adapter.py`: Harbor import-path compatible adapter shape.

## AReaL

- `areal_adapter.py`: async `run(data, **extra_kwargs)` wrapper that returns scalar reward.

## Batch execution

Use either the module entrypoint or installed script:

```bash
uv run --directory cli python -m benchmark \
  --config cli/config.json \
  --agent terminus-2 \
  --model qwen3-coder-next \
  --input-jsonl /path/to/tasks.jsonl \
  --output-jsonl /path/to/results.jsonl
```

```bash
uv run --directory cli small-agent-benchmark \
  --config cli/config.json \
  --agent terminus-2 \
  --model qwen3-coder-next \
  --input-jsonl /path/to/tasks.jsonl \
  --output-jsonl /path/to/results.jsonl
```

## Terminal-Bench 2.0 (Harbor harness) integration

This repo includes an additive import-path bridge at
`benchmark.harbor_bridge:HarborTB2DefaultAgent`.

- It resolves `default_agent` and `default_model` from `cli/config.json`.
- You can override with `--agent-kwarg agent_key=<key>` and
  `--agent-kwarg model_key=<key>`.

Smoke check (single task):

```bash
uvx --from terminal-bench tb run \
  -d terminal-bench@2.0 \
  --agent-import-path benchmark.harbor_bridge:HarborTB2DefaultAgent \
  --agent-kwarg config_path=./config.json \
  --task-id hello-world \
  --n-attempts 1 \
  --n-concurrent 1
```

Small concurrent run (10 tasks):

```bash
uvx --from terminal-bench tb run \
  -d terminal-bench@2.0 \
  --agent-import-path benchmark.harbor_bridge:HarborTB2DefaultAgent \
  --agent-kwarg config_path=./config.json \
  --n-tasks 10 \
  --n-attempts 1 \
  --n-concurrent 4
```

Prerequisites:

- Docker running locally.
- Model API keys exported for the selected default model
  (for `qwen3-coder-next`, set `OPENROUTER_API_KEY`).

