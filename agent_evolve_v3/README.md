# Agent Evolve V3

`agent_evolve_v3` is a self-contained outer loop for evolving benchmark
workspaces from the `terminus2` harness with a separate planner and
implementation agent.

## Key ideas

- The starting harness is selected in `runs.json`.
- Each state is a full workspace snapshot with visible runtime, benchmark,
  validation, and documentation files.
- A planning agent reviews all completed states and selects which parent state
  to branch from next.
- A separate implementation agent executes the planner's single selected change
  in a copied child workspace.
- Each child state inherits the selected parent's official benchmark artifacts
  for reference, then gets exactly one automatic official benchmark after
  implementation.
- The planner consumes the persisted official benchmark results from prior
  states on the next planning phase.

## Start a run

```bash
uv run python agent_evolve_v3/run_outer_loop.py --run-name terminus2-default
```

## Resume a run

```bash
uv run python agent_evolve_v3/run_outer_loop.py --resume agent_evolve_v3/outputs/<run-dir>
```

## What gets created

Each run writes:

- `run_manifest.json` with the selected baseline and model settings
- `states/iteration-XXXX.json` for persisted tree nodes
- `workspaces/iteration-XXXX/` for self-contained editable workspaces
- `artifacts/iteration-XXXX/` for planner, implementation, validation, and
  benchmark outputs
- `SCOREBOARD.md` with per-state metrics

## Workspace flow

Every materialized workspace contains:

- the copied selected harness under `agent/`
- workspace-local runtime and Harbor entrypoints
- an optional smoke benchmark launcher plus copied prior benchmark artifacts
- validation scripts and tests
- `README.md`

The checked-in seeds under `agent_evolve_v3/start_workdirs/` are the actual
pre-assembled starting workspaces.

The outer loop only coordinates these workspaces. The workspaces themselves are
intended to be understandable and runnable without referring back to the older
`agent_evolve` implementation.

## Dashboard

A live-updating web dashboard for monitoring ongoing runs.

```bash
uv run python -m agent_evolve_v3.dashboard
```

Then open http://localhost:8000 in a browser.
