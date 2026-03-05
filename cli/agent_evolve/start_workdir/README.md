# Agent Evolve Workdir

This directory is the self-contained workspace used by the headless CLI agent.
All code changes should stay inside this workdir unless explicitly required.

## Path convention

- In this folder, files are referenced directly (for example, `agent.py`).
- When this folder is copied into an iteration workspace, it is mounted as `agent_evolve/`.
  In that context, the same file is addressed as `agent_evolve/agent.py`.

## Quickstart

- Run benchmark and capture artifacts:
  `uv run python agent_evolve/run_recorded_benchmark.py --iteration 1 --runner cli/harbor/run_small.sh`
- Run interface contract tests:
  `uv run python -m unittest agent_evolve.test_interface_contract`

## Artifacts

- Eval artifacts are written under `eval/iter-<NNNN>/run-<NNNN>/`.
- Code snapshots are written under `snapshots/iter-<NNNN>/run-<NNNN>/`.
- Each benchmark run writes an `eval_summary.json` in its eval run directory.

## File map

- `agent.py`  
  Single-file agent implementation: runtime adapter plus core terminal-driving loop,
  including model calls, JSON parsing, command execution, and control flow.

- `../headless_inner_loop_prompt.md`  
  Prompt template used by `run_outer_loop.py` to drive each inner-loop improvement step.

- `run_recorded_benchmark.py`  
  Benchmark entrypoint for this workdir. Runs Harbor benchmark, stores eval artifacts, and snapshots code.

- `test_interface_contract.py`  
  Interface compatibility and benchmark contract tests for this workdir agent.

- `NOTES.md`  
  Working notes file. Record observations, hypotheses, changes, and outcome.

- `.gitignore`  
  Ignore rules for generated artifacts and caches.

- `__init__.py`  
  Package marker.
