# Workspace README

This workspace was seeded from the `{baseline}` baseline harness.

## Layout

- `agent/` contains the entire editable harness.
- `agent/agent.py` is the main agent entrypoint.
- `agent/runtime_types.py` defines the runtime contract used by the hidden services.
- `test_agent.sh` smoke-tests the local harness against the hidden validation service.
- `run_benchmark.sh` sends the local harness to the hidden benchmark service.
- `outputs/` holds only the benchmark runs visible from this workspace.
- `NOTES.md` is the working notebook for the refiner.

## Run validation

```bash
./test_agent.sh {model_key}
```

## Run eval

```bash
./run_benchmark.sh {model_key}
```

## Cache behavior

The benchmark cache lives outside this workspace. Re-running the benchmark on
unchanged `agent/` code may reuse a cached canonical result while still
recording a new visible local run in `outputs/`.

Only files under `agent/` participate in benchmark invalidation. In practice:

- Python files under `agent/` are hash-relevant.
- Prompt markdown used by Python code under `agent/` is hash-relevant.
- `README.md`, `NOTES.md`, wrapper scripts, and visible outputs do not trigger a
  new benchmark run.

While the refiner is actively working, multiple visible benchmark runs may
accumulate under `outputs/`. When a new refiner or critic workspace is copied,
its `outputs/` is trimmed back to exactly one visible official run for the
starting code state.

## Ground rules

- Keep changes general-purpose.
- Do not reach back into the old `agent_evolve`, `cli.py`, or prior benchmark
  wrapper codepaths.
- Validation and benchmarking are provided by hidden local services; there is no
  workdir-local critic command.
- Update `NOTES.md` with your hypothesis, validation results, and whether the
  change should generalize.
