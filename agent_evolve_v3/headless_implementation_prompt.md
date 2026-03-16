You are improving a self-contained benchmark workspace for `agent_evolve_v3`.

The selected baseline harness for this run is `{baseline}`.
You are refining a child workspace cloned from parent iteration **{parent_iteration}**.

Workspace root: `{workspace_root}`
Selected parent workspace (reference only, do not edit there): `{parent_workspace}`
Run root: `{run_root}`

This loop uses planner-driven branching:
- A planning agent reviewed all completed states and selected this parent.
- You must implement exactly one change: the planner-selected plan below.
- The outer loop will run exactly one official full benchmark after your implementation is done.

Current best completed state:
- iteration: `{best_iteration}`
- reward: `{best_reward}`
- passed / failed / errors: `{best_passed}` / `{best_failed}` / `{best_errors}`
- workspace (reference only): `{best_workspace}`

Selected parent state's official benchmark:
- reward: `{parent_reward}`
- passed / failed / errors: `{parent_passed}` / `{parent_failed}` / `{parent_errors}`
- summary json: `{parent_benchmark_summary_path}`
- stdout log: `{parent_benchmark_stdout_path}`
- stderr log: `{parent_benchmark_stderr_path}`
- Harbor job dir: `{parent_harbor_job_dir}`

{scoreboard}

## Planner-selected plan

{plan}

## Workspace rules

- The workspace is intentionally minimal. Edit only the local `agent/` harness code plus workspace docs/notes.
- Treat `{workspace_root}` as the only editable workspace for this iteration.
- Do not edit files inside `{parent_workspace}`. That parent workspace is reference context only.
- Compare your final change against the best completed state and avoid obvious regressions in its stronger behavior.
- Read `README.md` first, then `NOTES.md`, then the copied benchmark artifacts under `outputs/` if the plan references prior failure modes.
- When you run the workspace-local validation or smoke benchmark wrapper, use model key `{benchmark_model_key}`.
- The selected parent's official benchmark artifacts were copied into `outputs/` for local reference.
- Do not run the full benchmark yourself unless you are intentionally doing extra manual investigation beyond the required flow.
- Keep changes general. Do not add task-name-specific hacks.
- Prefer a small, well-documented change over a wide rewrite unless the planner evidence clearly points to an architectural issue.

## Required validation flow

Before you finish:
1. Run the workspace validation command described in `README.md`.
2. Run the optional smoke benchmark command from `README.md` only if you need a quick local benchmark check.
3. Update `NOTES.md` with your hypothesis, the change you made, and whether you expect it to generalize.

After you finish, the outer loop will automatically run the official benchmark and store those results for the next planning phase.

## Goal

Improve benchmark performance on the configured task set by refining the copied harness code under `agent/`.
