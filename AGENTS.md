# AGENTS.md

## Tooling

- Use `uv` for Python commands and dependency management (for example: `uv run ...`, `uv add ...`).
- Format code with `ruff format` before finishing changes.
- Run lint checks with `ruff check` before finishing changes.
- Run type checks with `basedpyright` before finishing changes (`uvx basedpyright`).

## Smoke Tests

Run these after any significant change to verify end-to-end correctness.

### Unit tests

```sh
uv run pytest --ignore=tests/test_agent_evolve_v2.py --ignore=tests/test_outer_loop.py --ignore=tests/test_recorded_benchmark.py -q
```

The three ignored test files reference deleted modules (`agent_evolve_v2`, `agent_evolve`) and are expected to fail at collection.

### CLI smoke test

```sh
uv run python -m cli --model gpt-5.4-nano-medium --max-turns 5 "echo hello"
```

Verifies the agent starts, sends the command, sees output, and exits cleanly with code 0.

### Workspace agent validation

```sh
bash agent_evolve_v3/start_workdirs/terminus2/test_agent.sh gpt-5.4-nano-medium
```

Loads the workspace agent, checks that `WorkspaceAgent` and `run_task` are available, and validates model key resolution. Exits with code 0 on success.

### Harbor smoke test (requires Docker + Harbor CLI)

```sh
./harbor/run_smoke.sh --model gpt-5.4-nano-medium
```

Runs a single benchmark task (`fix-git`) in a Docker sandbox. Requires Docker to be running and the Harbor CLI to be installed.

### Agent evolution outer loop smoke test (expensive)

```sh
uv run python -m agent_evolve_v3.run_outer_loop --run-name terminus2-smoke
```

Runs the full agent evolution pipeline end-to-end: spins up Docker containers, calls LLMs for planning (`gemini-3.1-pro`) and failure investigation (`gemini-3-flash`), benchmarks the agent (`gpt-5.4-nano-medium`), and iterates for 2 rounds. This is slow and costs real money in API calls. Only run when validating changes to the evolution framework itself.

## Style Guidelines

### `if` / `else` spacing

- Do not add a blank line between an `if` block and its matching `else`.
- Add one blank line after an `else` block ends.
- For `if` blocks with no `else`, add one blank line after the `if` block ends.

#### Good

```python
if ready:
    start()
else:
    wait()

next_step()
```

```python
if ready:
    start()

next_step()
```

#### Bad

```python
if ready:
    start()

else:
    wait()
next_step()
```

```python
if ready:
    start()
next_step()
```

### Multi-argument function calls must use kwargs

- When a function call has multiple arguments, pass them as named keyword arguments.
- This applies to callbacks and helper calls like the `_render_response(...)` call near `cli.py`.

#### Good

```python
on_reasoning=lambda turn, parsed: _render_response(
    console=console,
    turn=turn,
    parsed=parsed,
    verbosity=args.verbosity,
)
```

#### Bad

```python
on_reasoning=lambda turn, parsed: _render_response(
    console,
    turn,
    parsed,
    args.verbosity,
)
```
