# Architecture constraints

This workspace uses the **OpenCode SDK** via `agent/orchestrator.ts` to run an agentic coding loop. The Python wrapper `agent/agent.py` invokes the orchestrator as a subprocess using `bun`.

## Do NOT

- **Do NOT delete or replace `agent/orchestrator.ts`**. It is the core of the agent and must remain.
- **Do NOT rewrite `agent/agent.py` to use litellm, OpenAI SDK, or any other LLM client directly**. The Python wrapper must delegate to the TypeScript orchestrator via subprocess.
- **Do NOT remove the `bun run orchestrator.ts` subprocess call** from `agent/agent.py`.

## What you CAN change

- **`agent/orchestrator.ts`**: Customize the OpenCode SDK config passed to `createOpencodeServer`. You can add custom agents, change prompts, add tools, adjust permissions, add MCP servers, or implement multi-step flows. See `OPENCODE_NOTES.md` for the full reference.
- **`agent/agent.py`**: Adjust how the Python wrapper parses the trajectory, handles errors, or passes environment variables. Keep the subprocess-to-bun architecture intact.

## How the agent works

1. `agent/agent.py` receives an instruction and runtime config.
2. It maps the model ID to an OpenCode-compatible provider/model string based on the API base URL (e.g. `openai/gpt-5.4-nano-2026-03-17` for OpenAI, `openrouter/<model>` for OpenRouter).
3. It invokes `bun run agent/orchestrator.ts <instruction>` as a subprocess, passing `OPENCODE_MODEL` and `OPENAI_API_KEY` via environment variables. The subprocess CWD is set to the task directory.
4. `orchestrator.ts` starts an OpenCode server with dynamic provider configuration, creates a session, sends the prompt, waits for completion via SSE events, extracts the trajectory, and prints it as JSON to stdout.
5. `agent/agent.py` parses the JSON trajectory and emits structured output.

## Improvement strategies

- Add custom agents with specialized prompts (e.g. a planner agent, a debugger agent).
- Implement multi-step flows in `orchestrator.ts` (plan → implement → verify).
- Customize tool permissions per agent.
- Add instruction files or system prompt overrides.
- Tune temperature, steps, or model parameters.
