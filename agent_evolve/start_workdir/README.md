# Agent Evolve Workdir

Self-contained workspace for the headless agent-evolve loop.
All code changes should stay inside this directory unless explicitly required.

## Quickstart

Run benchmark and capture artifacts:

```
uv run python agent_evolve/run_recorded_benchmark.py --iteration 1 --runner harbor/run_dev_benchmark.sh
```

Run interface contract tests:

```
uv run python -m unittest agent_evolve.test_interface_contract
```
