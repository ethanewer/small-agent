# Agent Evolve

Automated pipeline that iteratively improves an inner agent's harness by running
benchmarks, invoking a Cursor agent to make improvements, validating the result,
and measuring generalization on a held-out eval set.

## Quick start

Start a fresh run with default settings:

```
uv run python cli/agent_evolve/run_outer_loop.py
```

## Resume a stopped run

If the pipeline was interrupted (crash, disconnect, Ctrl-C), resume from exactly
where it left off:

```
uv run python cli/agent_evolve/run_outer_loop.py --resume cli/agent_evolve/outputs/run-YYYYMMDDTHHMMSSZ
```

The `--resume` flag reuses the existing run directory and reads `run_state.json`
to determine which iteration and step to continue from. No work is repeated.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations N` | `25` | Total number of outer loop iterations to run |
| `--start-iteration N` | `1` | Starting iteration (ignored on resume; state file takes precedence) |
| `--runner PATH` | `cli/harbor/run_debug.sh` | Dev benchmark runner script (5-task debug split) |
| `--eval-runner PATH` | `cli/harbor/run_small_benchmark.sh` | Eval benchmark runner (15 tasks, run between iterations) |
| `--agent-key KEY` | `terminus-2` | Agent key passed to Harbor |
| `--model-key KEY` | *(from config.json)* | Model key override |
| `--cursor-model MODEL` | *(default)* | Model for the Cursor agent (inner loop) |
| `--resume PATH` | *(none)* | Path to existing run directory to resume |
| `--skip-initial-benchmark` | `false` | Skip iteration-1 dev benchmark if a cached baseline exists |
| `--runner-args STR` | *(none)* | Extra arguments forwarded to the dev runner (e.g. `"--split 1"`) |

## Pipeline flow

Each outer iteration runs these steps in order:

```
1. Dev benchmark    Run 5-task debug split (terminal-bench@2.0 subset).
                    Inner agent sees these results.
                    Non-fatal: failure is recorded, pipeline continues.

2. Cursor agent     Reads dev results, modifies agent_evolve/agent.py.
                    A pre-cursor snapshot is saved first for rollback.
                    Fatal: if cursor fails, iteration stops.

3. Validation       Runs test_interface_contract to verify agent.py.
                    Fatal: if tests fail, iteration stops.

4. Eval benchmark   Run 15-task benchmark (terminal-bench@2.0 subset).
                    Run between outer loop iterations.
                    Skipped if agent.py is unchanged since last eval.
                    Non-fatal: failure is recorded, pipeline continues.

5. State save       Records iteration completion, eval score, agent hash.
```

## Datasets

Subsets of `terminal-bench@2.0`:

- **Dev set** (5 tasks per split): 4 splits of 5 medium tasks each, run via
  `run_debug.sh --split N`. Used by the inner agent to diagnose failures.
- **Eval set** (15 tasks): 4 easy + 10 medium + 1 hard, run via `run_small_benchmark.sh`.
  Run between outer loop iterations. Disjoint from the dev set.
- **Full set** (89 tasks): All tasks, run via `run_full_benchmark.sh`.

## Run directory layout

```
outputs/run-YYYYMMDDTHHMMSSZ/
  run_state.json              Pipeline state (iteration, step, scores)
  agent_evolve/               Working copy of the inner agent code
    agent.py                  The file being evolved
    run_recorded_benchmark.py Benchmark runner
    test_interface_contract.py Validation tests
    NOTES.md                  Inner agent's working notes
    README.md                 Workdir documentation
  eval/
    latest_run.json           Most recent dev benchmark summary
    latest_eval.json          Most recent eval benchmark summary
    iter-0001/
      outer_benchmark_step.json
      outer_cursor_step.json
      outer_validation_step.json
      outer_eval_benchmark_step.json
      cursor_prompt.txt
      run-0001/               Dev benchmark artifacts
        eval_summary.json
        harbor_job/           Copied Harbor job directory
      eval-0001/              Eval benchmark artifacts
        eval_summary.json
        harbor_job/
  snapshots/
    iter-0001/
      pre-cursor/             Code snapshot before cursor agent ran
      run-0001/               Code snapshot at dev benchmark time
```

## State file (`run_state.json`)

| Field | Description |
|-------|-------------|
| `current_iteration` | The iteration currently in progress (or next to run) |
| `last_completed_step` | Last step completed in current iteration: `dev_benchmark`, `cursor`, `validation`, `eval`, or `null` |
| `eval_score` | Most recent eval benchmark reward mean |
| `last_eval_agent_hash` | SHA-256 of agent.py at last eval (used to skip unchanged evals) |
| `dev_benchmark_failed` | Whether the dev benchmark failed this iteration |
| `eval_benchmark_failed` | Whether the eval benchmark failed this iteration |
| `stop_requested` | Whether a graceful stop was requested (SIGINT/SIGTERM) |
| `runner` | Dev benchmark runner path |
| `eval_runner` | Eval benchmark runner path |
| `agent_key` | Harbor agent key |
| `model_key` | Model key override (or null) |

## Troubleshooting

**Stale lock file**: If a run was killed and you see "Another benchmark run
appears to be active", the lock file at `cli/harbor/jobs/.agent_evolve_benchmark.lock`
is stale. The pipeline auto-detects dead PIDs and cleans up, but if the PID was
recycled, manually delete the lock file.

**Check if a run is active**: Read `run_state.json` in the run directory. If
`last_completed_step` is non-null, the run was interrupted mid-iteration.
Use `--resume` to continue.

**Manually advance past a stuck iteration**: Edit `run_state.json`, set
`current_iteration` to the desired value and `last_completed_step` to `null`,
then resume.

**Rollback a bad cursor change**: Copy the pre-cursor snapshot back:
```
cp -r snapshots/iter-NNNN/pre-cursor/* agent_evolve/
```
Then resume -- the pipeline will re-run from the cursor step.
