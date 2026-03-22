# Workspace README

This workspace was seeded from the `{baseline}` baseline harness.

## Architecture

**IMPORTANT**: This agent uses the OpenCode SDK (`@opencode-ai/sdk/v2`) via a TypeScript orchestrator. Do NOT replace this architecture with litellm, the OpenAI SDK, or any other direct LLM client. Read `AGENTS.md` for the full list of constraints.

The agent flow is:

1. `agent/agent.py` (Python wrapper) → spawns `bun run agent/orchestrator.ts`
2. `agent/orchestrator.ts` (TypeScript) → uses the OpenCode SDK to run an agentic loop with tools (bash, read, write, edit, grep, glob, etc.)
3. The orchestrator returns a JSON trajectory on stdout.

## Layout

- `agent/agent.py` — Python wrapper that invokes the orchestrator. **Keep the subprocess-to-bun architecture.**
- `agent/orchestrator.ts` — OpenCode SDK orchestrator. **This is the core of the agent. Do NOT delete it.**
- `agent/runtime_types.py` — Runtime contract used by the hidden services.
- `AGENTS.md` — Architecture constraints and improvement strategies. **Read this first.**
- `OPENCODE_NOTES.md` — Full reference for the OpenCode SDK: agents, tools, prompts, orchestration patterns.
- `test_agent.sh` — Smoke-tests the local harness against the hidden validation service.
- `run_smoke_benchmark.sh` — Runs the single-task smoke benchmark for quick checks.
- `outputs/` — Copied official benchmark artifacts from the selected parent state.
- `outputs/trajectory.json` — Full agent trajectory from the last run.

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
- Do not reach back into the old `agent_evolve`, `cli.py`, or prior benchmark wrapper codepaths.
- Validation and benchmarking are provided by hidden local services.
- Read the copied prior benchmark artifacts under `outputs/` when the plan references specific failure modes.
- Read `OPENCODE_NOTES.md` for reference on how to customize agents, tools, prompts, and orchestration via the OpenCode SDK.
- **All improvements must work through the OpenCode SDK orchestrator.** Do not bypass it.
