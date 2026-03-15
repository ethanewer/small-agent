You are improving a self-contained benchmark workspace for `agent_evolve_v2`.

The selected baseline harness for this run is `{baseline}`.
You are refining a child workspace cloned from parent iteration **{parent_iteration}**.

Workspace root: `{workspace_root}`
Selected parent workspace (reference only, do not edit there): `{parent_workspace}`
Run root: `{run_root}`

This loop uses paper-aligned tree search:
- Parent selection is Thompson-sampled in code, not chosen by a planner agent.
- There is one benchmark pass per child state.
- The benchmark output becomes both the score and the input to the critic.
- You should make one generalizable change per iteration.

Current Thompson sample:
- sample: `{thompson_sample}`
- alpha: `{alpha}`
- beta: `{beta}`
- workspace benchmark model key: `{benchmark_model_key}`

{scoreboard}

## Parent critic summary

{critic_summary}

## Workspace rules

- The workspace is intentionally minimal. Edit only the local `agent/` harness code plus workspace docs/notes.
- Treat `{workspace_root}` as the only editable workspace for this iteration.
- Do not edit files inside `{parent_workspace}`. That parent workspace is reference context only.
- Do not reach back into the repo's older `agent_evolve`, `cli.py`, `harbor/agent.py`, or v1 benchmark wrapper codepaths.
- Read `README.md` first, then `NOTES.md`, then any benchmark artifacts referenced there.
- When you run the workspace-local validation or benchmark wrapper, use model key `{benchmark_model_key}`.
- Visible benchmark outputs live under `outputs/`. While you work, multiple local benchmark outputs may appear there.
- Benchmark results are deduplicated by a shared cache, so rerunning the benchmark on unchanged code may reuse an existing canonical result.
- Only files under `agent/` participate in benchmark invalidation; edits to `README.md` or `NOTES.md` do not trigger a fresh benchmark.
- Keep changes general. Do not add task-name-specific hacks.
- Prefer small, well-documented changes over wide rewrites unless the critic clearly points to an architectural issue.

## Required validation flow

Before you finish:
1. Run the workspace validation command described in `README.md`.
2. Run the single benchmark/eval command described in `README.md`.
3. Update `NOTES.md` with your hypothesis, the change you made, and whether you expect it to generalize.

## Goal

Improve benchmark performance on the single `run_small_benchmark.sh` task set by refining the copied harness code under `agent/`.
