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

## Terminal-Bench integration (official registry tasks)

This repo exposes a TB import-path bridge:
`benchmark.harbor_bridge:HarborTB2DefaultAgent`

For a full command cookbook, see:
- `benchmark/examples/README.md`

- Uses defaults from `cli/config.json` (`default_agent`, `default_model`)
- Optional overrides:
  - `--agent-kwarg config_path=...`
  - `--agent-kwarg agent_key=...`
  - `--agent-kwarg model_key=...`
- The Harbor bridge delegates to official terminal-bench agents, so task
  commands execute through the provided tmux session in the docker sandbox.

Smoke check (single task from the official task registry):

```bash
uvx --with pexpect --with rich --from terminal-bench tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path benchmark.harbor_bridge:HarborTB2DefaultAgent \
  --agent-kwarg config_path="$(pwd)/cli/config.json" \
  --task-id fix-git \
  --n-attempts 1 \
  --n-concurrent 1 \
  --output-path "${TMPDIR:-/tmp}/small-agent-tb2-runs"
```

Tiny subset run (5 tasks, all present in Terminal-Bench 2.0 and terminal-bench-core==0.1.1):

```bash
uvx --with pexpect --with rich --from terminal-bench tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path benchmark.harbor_bridge:HarborTB2DefaultAgent \
  --agent-kwarg config_path="$(pwd)/cli/config.json" \
  --task-id configure-git-webserver \
  --task-id fix-git \
  --task-id count-dataset-tokens \
  --task-id sqlite-db-truncate \
  --task-id nginx-request-logging \
  --n-attempts 1 \
  --n-concurrent 4 \
  --output-path "${TMPDIR:-/tmp}/small-agent-tb2-runs"
```

