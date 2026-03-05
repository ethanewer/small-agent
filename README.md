# small-agent

This repository contains two areas:

- `cli/` -- active CLI codebase (Terminus-2 wrapper)
- `sft/` -- legacy SFT dataset and training pipeline

## CLI

The CLI provides a local wrapper around the Terminus-2 JSON interaction pattern.
It runs commands in a persistent local shell and formats output with Rich.

### Requirements

- Python 3.13+
- API key(s) exported in your shell for any providers you configure:

```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

Each model can reference its own env var in config.

### Configuration

Model settings live under `models` in `cli/config.json`.
This file is tracked, so prefer env-var references for `api_key` values instead of literal secrets.

```json
{
  "default_agent": "terminus-2",
  "default_model": "qwen3-coder-next",
  "agents": {
    "terminus-2": {
      "final_message": false
    },
    "qwen": {}
  },
  "models": {
    "qwen3-coder-next": {
      "model": "qwen/qwen3-coder-next",
      "api_base": "https://openrouter.ai/api/v1",
      "api_key": "OPENROUTER_API_KEY"
    },
    "gpt-5.3-codex": {
      "model": "gpt-5.3-codex",
      "api_base": "https://api.openai.com/v1",
      "api_key": "OPENAI_API_KEY"
    }
  },
  "verbosity": 0,
  "max_turns": 50,
  "max_wait_seconds": 60
}
```

- `default_agent`: default agent key to use from `agents`
- `default_model`: default model key to use from `models`
- `agents`: dict of named agent profiles/options
- `models`: dict of named model profiles
- model profile `model`: provider model ID (for example `gpt-5.3-codex`)
- model profile `api_base`: provider base URL
- model profile `api_key`: literal key, env var name (`OPENAI_API_KEY`), or `$ENV_VAR`
- model profile `temperature` (optional): sampling temperature for that model; if omitted, provider defaults are used
- `verbosity`: `0` (compact) or `1` (full tool IO + reasoning)
- `max_turns`: maximum model turns before stopping
- `max_wait_seconds`: max time to wait for each tool call/command completion

### Install

```bash
cd cli
./setup
```

This installs agent CLIs locally under `cli/.local/bin` and `cli/.local/tools` (no global npm/pipx installs).

Add local wrappers to your shell path:

```bash
export PATH="$PWD/.local/bin:$PATH"
```

To remove all locally installed CLI artifacts:

```bash
./clean
```

### Usage

Run with a positional instruction:

```bash
./.local/bin/terminus2-cli "List files in current directory, then explain what you see."
```

Or run without instruction and enter it interactively when prompted:

```bash
./.local/bin/terminus2-cli
```

Optional flags:

```bash
./.local/bin/terminus2-cli --verbosity 1 --max-turns 10 --model gpt-5.3-codex --config ./config.json "Your instruction"
```

- `--model <key>` selects a model key from `config.models`.
- `--agent <key>` selects an agent key from `config.agents`.

### Available Agents

- `terminus-2`: interactive terminal-driving JSON agent
- `qwen`: runs `qwen -p "<instruction>" -y` with OpenAI-compatible env

### Harbor External Agent

You can run this project through Harbor without modifying Harbor source code by
importing the external adapter class:

```bash
harbor run --path "<task-or-dataset-path>" --agent-import-path agent:SmallAgentHarborAgent
```

The adapter module is `cli/harbor/agent.py`, and the scripts run Harbor from
that directory so `--agent-import-path agent:SmallAgentHarborAgent` resolves
correctly.

Runtime separation notes:

- Harbor wrapper runtime (in `cli/harbor/agent.py`) only loads lightweight
  config helpers from `cli/harbor_config.py`.
- Task runtime executes `python3 /tmp/small-agent-cli/cli.py ...` inside the
  benchmark environment and owns heavy CLI dependencies (`litellm`, `pexpect`,
  etc.).
- This split prevents Harbor wrapper dependency resolution from importing full
  task agent stacks.

The adapter resolves defaults from `cli/config.json`:

- agent default -> `default_agent`
- model default -> `default_model`

You can override either at runtime using Harbor agent env:

- `SMALL_AGENT_HARBOR_AGENT=<agent-key>`
- `SMALL_AGENT_HARBOR_MODEL=<model-key>`

Both keys are validated against `config.json` (`agents` and `models`).

### Harbor Runner Scripts

The CLI includes helper scripts with fixed public Terminal-Bench datasets:

- `cli/harbor/run_small.sh`: small dataset (`terminal-bench-sample@2.0`)
- `cli/harbor/run_full.sh`: full dataset (`terminal-bench@2.0`)

All scripts accept:

- `--model <key>`: selects a model key from `cli/config.json`
- `--agent <key>`: selects an agent key from `cli/config.json`
- `--dry-run`: prints resolved command without executing Harbor

If `--model`/`--agent` are omitted, scripts use config defaults.
Extra Harbor CLI arguments are intentionally rejected by these scripts.

Safety defaults enforced by the scripts:

- write job artifacts only under `cli/harbor/jobs/<run-id>/...` (`--jobs-dir` is fixed)
- force Docker environment execution (`--env docker`)
- force environment cleanup (`--delete`)
- disable forced image rebuilds (`--no-force-build`)

This keeps host-visible benchmark outputs scoped to `cli/harbor/jobs`, while
task execution remains isolated in Harbor-managed Docker environments.

Examples:

```bash
cd cli
./harbor/run_small.sh --model gpt-5.3-codex --agent qwen --dry-run
./harbor/run_full.sh --agent terminus-2
```

### Iterative Agent Evolver Pipeline

This repository now includes an overnight evolver loop in
`cli/agent_evolve/` that:

- starts from a stripped, core-only Terminus-2-derived agent
- runs benchmark/eval cycles against Harbor
- snapshots workdir code each benchmark run
- stores eval artifacts next to snapshots
- can stop gracefully (Ctrl+C/SIGTERM) after the in-flight step
- creates contained run workdirs under `cli/agent_evolve/outputs/`
- runs the inner Cursor agent with `--workspace <run-dir> --sandbox enabled`
  so evolution is contained to each run directory

Key entrypoints:

- outer loop: `cli/agent_evolve/run_outer_loop.py`
- starting template workdir: `cli/agent_evolve/start_workdir/`
- benchmark wrapper: `cli/agent_evolve/start_workdir/run_recorded_benchmark.py`
- inner-loop prompt template:
  `cli/agent_evolve/headless_inner_loop_prompt.md`
- interface compatibility tests:
  `cli/agent_evolve/start_workdir/test_interface_contract.py`

Default run (25 iterations, small benchmark):

```bash
cd cli
uv run python agent_evolve/run_outer_loop.py
```

Common overrides:

```bash
cd cli
uv run python agent_evolve/run_outer_loop.py --iterations 25 --runner cli/harbor/run_small.sh
uv run python agent_evolve/run_outer_loop.py --cursor-model gpt-5.3-codex
```

Artifacts:

- all outputs are under `cli/agent_evolve/outputs/<run-id>/`
- current run workdir: `cli/agent_evolve/outputs/<run-id>/agent_evolve/`
- eval logs/results: `cli/agent_evolve/outputs/<run-id>/eval/`
- agent code snapshots: `cli/agent_evolve/outputs/<run-id>/snapshots/`
- run state: `cli/agent_evolve/outputs/<run-id>/run_state.json`

`run_recorded_benchmark.py` is the required benchmark entrypoint for evolver runs.
It guarantees each benchmark invocation generates both a code snapshot and copied
Harbor eval artifacts.

### Interactive Commands

When entering instruction interactively, you can use slash commands before starting the run:

- `/model`: choose a model from a numbered list.
- `/agent`: choose an agent from a numbered list.
- `/verbosity [0|1]`: set runtime verbosity.
- `/max_turns [N]`: set runtime max turns (`N >= 1`).
- `/max_wait_seconds [S]`: set runtime max wait (`S > 0`).

If you omit a value (for example, `/verbosity`), the CLI prompts you for one.

Example:

```bash
./.local/bin/terminus2-cli
# Enter instruction: /model
# Available Models:
# 1. qwen3-coder-next (...)
# 2. gpt-5.3-codex (...)
# Enter model number: 2
# Enter instruction: /max_turns 20
# Enter instruction: /verbosity
# Enter verbosity (0 or 1): 1
# Enter instruction: Diagnose failing tests
```

### Verbosity Levels

- `0`: one line per tool call
- `1`: all tool call inputs/responses, plus intermediate reasoning output

### Headless Agent Environment

`qwen` relies on model profile values from `config.json`:

- `model` -> `OPENAI_MODEL`
- `api_base` -> `OPENAI_BASE_URL`
- `api_key` -> `OPENAI_API_KEY` (resolved from env var name or literal)

Compatibility behavior:

- OpenRouter and local OpenAI-compatible endpoints are first-class targets.
- OpenAI models are allowed when the selected agent/CLI supports the model type.
- Known incompatible combinations are surfaced with explicit agent errors.

Examples:

```bash
./.local/bin/terminus2-cli --agent qwen --model qwen3-coder-next "Summarize this repository"
```

You can override the executable per agent if needed:

```json
{
  "agents": {
    "qwen": { "binary": "qwen" }
  }
}
```

### Completion Message

By default, after `task_complete` is confirmed twice in a row, the CLI sends
one extra post-run prompt to the model (using the same chat history) to
generate a final user-facing summary message.

If that post-run summary call fails or returns empty text, the CLI falls back
to the optional `final_message` from the agent JSON response, and then to the
default completion text.

You can disable the final summary panel per agent:

```json
{
  "agents": {
    "terminus-2": {
      "final_message": false
    }
  }
}
```

## SFT Dataset Pipeline (Legacy)

The `sft/` directory builds combined SFT datasets from:

- `nvidia/Nemotron-Terminal-Corpus` (all terminal trajectory splits)
- `Nanbeige/ToolMind-Web-QA`
- `SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K`

The pipeline normalizes rows into a unified `messages` schema, removes
`<think>...</think>` reasoning, concatenates and shuffles deterministically,
and keeps data in `messages` format so training can mask user turns.

### SFT Setup

```bash
cd sft
uv sync
```

### Dataset Build Modes

**Demo** (100 rows/source for quick validation):

```bash
uv run qwen-dataset-build --mode demo --rows-per-source 100 \
  --demo-output-dir "/path/to/demo-dataset"
```

**Full** (all rows from all sources):

```bash
uv run qwen-dataset-build --mode full \
  --full-output-dir "/path/to/full-dataset"
```

**Balanced** (target ~50% NVIDIA bytes):

```bash
uv run qwen-dataset-build --mode balanced \
  --balanced-output-dir "/path/to/balanced-dataset"
```

**Additional Nemotron** (one-off, capped at 10k rows/split):

```bash
uv run qwen-dataset-build --mode additional \
  --additional-rows-per-split 10000 \
  --additional-output-dir "/path/to/additional-data"
```

**Extended full** (merge existing full + additional JSONL):

```bash
uv run qwen-dataset-build --mode extended_full \
  --base-full-input-jsonl "/path/to/full-dataset/dataset_shuffled.jsonl" \
  --additional-input-jsonl "/path/to/additional-data/dataset_shuffled.jsonl" \
  --extended-output-dir "/path/to/full-dataset-extended"
```

### Output Artifacts

Each dataset directory contains:

- `dataset/` -- Hugging Face dataset saved to disk
- `metadata.json` -- source counts/bytes, policy details, reproducibility fields
- `dataset_shuffled.jsonl` (full/balanced) -- deterministic shuffled JSONL

Demo mode also writes `examples.jsonl` (chat-template-rendered) and `source_samples.jsonl`.

### SFT Training (ms-swift)

Export demo data and submit a 2-node B200 Slurm job:

```bash
uv run python data/scripts/export_demo_for_swift.py \
  --input-dir data/demo-dataset/dataset \
  --output-file data/demo-dataset/train.jsonl

mkdir -p logs outputs
sbatch sft-scripts/train.sbatch
```

### Caveats

- Set `HF_TOKEN` for higher Hugging Face rate limits.
- No token-length filtering is done in preprocessing; filter at training time.
- Think-tag cleanup is applied before writing rows.
