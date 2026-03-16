# Workspace README

This workspace was seeded from the `{baseline}` baseline harness.

## Layout

- `agent/` contains the entire editable harness.
- `agent/agent.py` is the main agent entrypoint.
- `agent/runtime_types.py` defines the runtime contract used by the hidden services.
- `test_agent.sh` smoke-tests the local harness against the hidden validation service.
- `run_smoke_benchmark.sh` runs the single-task smoke benchmark for quick checks.
- `outputs/` holds the copied official benchmark artifacts from the selected parent state.
- `NOTES.md` is the working notebook for the refiner.

## Run validation

```bash
./test_agent.sh {model_key}
```

## Optional smoke benchmark

```bash
./run_smoke_benchmark.sh {model_key}
```

The full official benchmark is run automatically by the outer loop after the
implementation step completes. Use the smoke benchmark only when you need a
quick local signal before finishing.

## Ground rules

- Keep changes general-purpose.
- Do not reach back into the old `agent_evolve`, `cli.py`, or prior benchmark
  wrapper codepaths.
- Validation and benchmarking are provided by hidden local services.
- Read the copied prior benchmark artifacts under `outputs/` when the plan
  references specific failure modes.
- Update `NOTES.md` with your hypothesis, validation results, and whether the
  change should generalize.
