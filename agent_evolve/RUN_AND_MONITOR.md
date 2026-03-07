# Agent Evolve — Run & Monitor

## Objective

Run the agent evolve pipeline and monitor its progress over the full run. Write periodic status reports so progress, regressions, and wins are visible at a glance.

## 1. Start the pipeline

From the repo root (`small-agent/`), launch the outer loop:

```bash
uv run python agent_evolve/run_outer_loop.py
```

This runs up to 25 iterations by default. Each iteration:

1. **Dev benchmark** — runs a 5-task debug split (`harbor/run_debug.sh`).
2. **Cursor agent** — reads dev results and modifies `agent.py`.
3. **Validation** — runs `test_interface_contract` to verify the agent.
4. **Eval benchmark** — runs the 15-task held-out benchmark (`harbor/run_small_benchmark.sh`).

If the pipeline is interrupted, resume with:

```bash
uv run python agent_evolve/run_outer_loop.py --resume agent_evolve/outputs/<run-dir>
```

Monitor `run_state.json` inside the run directory to check which iteration and step the pipeline is on.

## 2. Reporting schedule

Write status reports on the following cadence:

| Time since start | Report frequency |
|-------------------|-----------------|
| 0 – 2 hours       | Every 30 minutes |
| 2+ hours           | Every 1 hour     |

Each report should be appended to a single file: `agent_evolve/outputs/<run-dir>/STATUS_REPORTS.md`, with a timestamp header.

## 3. Report contents

Every report must include the following three sections.

### 3a. Score figure (scatterplot)

Generate a scatterplot image (save as PNG in the run directory) with:

- **X-axis**: Benchmark run index, sorted by timestamp (earliest = 0, then 1, 2, …).
- **Y-axis**: Score (mean reward), range 0 to 1.
- **Colors**: Each benchmark type is a different color. Treat each debug split as its own benchmark. The benchmarks are:
  - `run_small_benchmark.sh` (eval, 15 tasks)
  - `run_debug.sh --split 1` (dev split 1, 5 tasks)
  - `run_debug.sh --split 2` (dev split 2, 5 tasks)
  - `run_debug.sh --split 3` (dev split 3, 5 tasks)
  - `run_debug.sh --split 4` (dev split 4, 5 tasks)
- **Baseline point**: Use `harbor/base_results/qwen3-coder-next-terminus-2-small-benchmark-result.json` as the first data point for `run_small_benchmark.sh` at x = 0. This file records a score of **0.40** (6/15 tasks passed).
- Include a legend mapping colors to benchmark names.

Data sources for the plot:
- `eval/iter-NNNN/run-*/eval_summary.json` — dev benchmark results (check `runner_args` or directory context to determine which split).
- `eval/iter-NNNN/eval-*/eval_summary.json` — eval benchmark results.
- `run_state.json` — current `eval_score` field for the latest eval.

### 3b. Recent agent activity

Summarize what the inner Cursor agent has been working on since the last report. To gather this information:

- Read the latest `eval/iter-NNNN/outer_cursor_step.json` files (the `stdout` field contains the agent's output).
- Read `agent_evolve/NOTES.md` inside the run's working directory for the agent's own notes.
- Diff `agent.py` against the previous iteration's snapshot (`snapshots/iter-NNNN/pre-cursor/agent.py`) to see what changed.

Summarize in 3–5 bullet points:
- What architectural changes were attempted.
- What robustness fixes were made.
- Whether the agent is exploring new approaches or refining an existing one.

### 3c. Best-performing harness features

Identify the iteration with the highest eval benchmark score so far and describe the features of that `agent.py`. To do this:

- Scan `eval/iter-NNNN/eval-*/eval_summary.json` for the highest `reward_mean`.
- Read the corresponding `agent.py` from `snapshots/iter-NNNN/pre-cursor/agent.py` (or the current working copy if it's the latest iteration).
- List the key features/architecture of that agent, for example:
  - Tool-use vs. raw keystrokes
  - Planning phase present or not
  - Scratchpad / persistent memory
  - Subagent delegation
  - Context summarization strategy
  - Error recovery mechanisms
  - Verification step before task completion
