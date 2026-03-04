# Benchmark Example Scripts

These scripts are runnable examples for a single model at three scales.

Run from anywhere; scripts resolve the repo automatically.

## Scripts

- `common.sh`: shared defaults and helper function.
- `run_smoke.sh`: 1-task smoke run for one model/agent.
- `run_tiny.sh`: 5-task tiny subset for one model/agent.
- `run_full.sh`: full dataset run for one model/agent.

## Usage

Smoke run (single model):

```bash
./cli/benchmark/examples/run_smoke.sh
```

Smoke run (explicit overrides):

```bash
./cli/benchmark/examples/run_smoke.sh --model qwen3-coder-next --agent terminus-2
```

Tiny subset for one model:

```bash
./cli/benchmark/examples/run_tiny.sh --model qwen3-coder-next --agent terminus-2 --concurrency 4
```

Tiny subset task IDs (official IDs shared by Terminal-Bench 2.0 and
`terminal-bench-core==0.1.1`):

- `configure-git-webserver`
- `fix-git`
- `count-dataset-tokens`
- `sqlite-db-truncate`
- `nginx-request-logging`

Full run for one model:

```bash
./cli/benchmark/examples/run_full.sh --model qwen3-coder-next --agent terminus-2 --concurrency 8 --attempts 1
```

## Environment overrides

All scripts support these optional env vars:

- `TB_DATASET` (default: `terminal-bench-core==0.1.1`)
- `TB_DATASET_PATH` (if set, uses `--dataset-path` directly; otherwise uses `--dataset`)
- `TB_OUTPUT_PATH` (default: `${TMPDIR:-/tmp}/small-agent-tb2-runs`)
- `TB_AGENT_IMPORT_PATH` (default: `benchmark.harbor_bridge:HarborTB2DefaultAgent`)
- `TB_CONFIG_PATH` (default: `./config.json`; resolved to an absolute path before launch)
- `TB_LOCAL_REGISTRY_PATH` (if set, passes `--local-registry-path`)
- `TB_USE_DATASET_CACHE` (default: `1`; when enabled and `TB_DATASET=name==version`,
  auto-uses `~/.cache/terminal-bench/<name>/<version>` if present)

If `--model` or `--agent` are omitted, scripts read `default_model` and
`default_agent` from `TB_CONFIG_PATH`.

## Reproducibility notes

- Benchmark scripts do not auto-switch to host cache paths. This avoids
  machine-specific `~/.cache/...` leakage in run metadata.
- Benchmark outputs default to a temp directory (not the git repo), reducing
  accidental workspace mutations during smoke/tiny/full runs.
- If your environment cannot fetch the remote registry (for example SSL/cert
  interception), the default cache fallback allows offline runs when the
  dataset version is already present under `~/.cache/terminal-bench/`.
- Task git operations must happen inside the Terminal-Bench sandbox. If you see
  host branch changes after a run, stop and inspect your agent import path.
