# Agent Evolve V2

`agent_evolve_v2` is a self-contained outer loop for evolving benchmark workspaces
from either the `liteforge` or `terminus2` harness.

## Key ideas

- The starting harness is selected in `runs.json`.
- Each state is a full workspace snapshot with visible runtime, benchmark, critic,
  validation, and documentation files.
- Parent selection uses Thompson sampling over prior benchmark success/failure
  counts.
- Each child state gets exactly one benchmark pass, then a critic summary is
  generated from those artifacts.

## Start a run

```bash
uv run python agent_evolve_v2/run_outer_loop.py --run-name liteforge-default
```

Use the `terminus2-default` profile to start from the `terminus2` baseline:

```bash
uv run python agent_evolve_v2/run_outer_loop.py --run-name terminus2-default
```

## Resume a run

```bash
uv run python agent_evolve_v2/run_outer_loop.py --resume agent_evolve_v2/outputs/<run-dir>
```

## What gets created

Each run writes:

- `run_manifest.json` with the selected baseline and model settings
- `states/iteration-XXXX.json` for persisted tree nodes
- `workspaces/iteration-XXXX/` for self-contained editable workspaces
- `artifacts/iteration-XXXX/` for cursor, validation, benchmark, and critic outputs
- `SCOREBOARD.md` with per-state metrics

## Workspace flow

Every materialized workspace contains:

- the copied selected harness under `agents/`
- workspace-local runtime and Harbor entrypoints
- a visible benchmark launcher
- validation scripts and tests
- `README.md` and `NOTES.md`

The checked-in seeds under `agent_evolve_v2/start_workdirs/` are the actual
pre-assembled starting workspaces. Refresh them after support or baseline code
changes with:

```bash
uv run python agent_evolve_v2/refresh_start_workdirs.py
```

The outer loop only coordinates these workspaces. The workspaces themselves are
intended to be understandable and runnable without referring back to the older
`agent_evolve` implementation.
