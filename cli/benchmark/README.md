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

