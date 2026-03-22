# Customizing and Orchestrating OpenCode Agents

This document covers every mechanism for customizing agents, controlling tools, replacing prompts, and orchestrating multi-step workflows — all without modifying source code.

## Table of Contents

- [Agents](#agents)
  - [Built-in Agents](#built-in-agents)
  - [Creating Agents via Markdown](#creating-agents-via-markdown)
  - [Creating Agents via JSON](#creating-agents-via-json)
  - [Agent Options Reference](#agent-options-reference)
- [Prompts](#prompts)
  - [Replacing the System Prompt](#replacing-the-system-prompt)
  - [Instruction Files](#instruction-files)
  - [Per-Call System Injection](#per-call-system-injection)
  - [Prompt Assembly Order](#prompt-assembly-order)
- [Tools](#tools)
  - [Built-in Tools](#built-in-tools)
  - [Disabling Tools via Permissions](#disabling-tools-via-permissions)
  - [Disabling Tools via the tools Key](#disabling-tools-via-the-tools-key)
  - [Per-Call Tool Overrides](#per-call-tool-overrides)
  - [Custom Tools via MCP](#custom-tools-via-mcp)
  - [Custom Tools via Tool Files](#custom-tools-via-tool-files)
  - [Custom Tools via Plugins](#custom-tools-via-plugins)
- [Orchestration](#orchestration)
  - [Headless CLI: opencode run](#headless-cli-opencode-run)
  - [Server Mode: opencode serve](#server-mode-opencode-serve)
  - [SDK Orchestration](#sdk-orchestration)
  - [Multi-Step Flows](#multi-step-flows)
  - [Extracting Trajectories](#extracting-trajectories)
  - [Subagent Invocation](#subagent-invocation)
  - [Event Stream](#event-stream)
- [Plugins](#plugins)
  - [Plugin Hooks Reference](#plugin-hooks-reference)
- [Configuration Precedence](#configuration-precedence)
- [Complete Example](#complete-example)

---

## Agents

### Built-in Agents

OpenCode ships with these agents:

| Agent | Mode | Description |
|-------|------|-------------|
| `build` | primary | Default agent with all tools enabled |
| `plan` | primary | Read-only analysis; edit denied by default, bash allowed |
| `general` | subagent | Full tool access for multi-step tasks |
| `explore` | subagent | Read-only codebase exploration |
| `compaction` | hidden | Auto-compacts long context |
| `title` | hidden | Generates session titles |
| `summary` | hidden | Creates session summaries |

Primary agents are cycled with **Tab** in the TUI. Subagents are invoked via `@name` mentions or automatically by the `task` tool.

### Creating Agents via Markdown

Place `.md` files in `.opencode/agents/` (project) or `~/.config/opencode/agents/` (global). The filename becomes the agent name.

```markdown
---
description: Security auditor that only reads code
mode: subagent
model: anthropic/claude-sonnet-4-20250514
temperature: 0.1
permission:
  edit: deny
  bash:
    "*": deny
    "git log*": allow
    "grep *": allow
  webfetch: deny
---

You are a security auditor. Analyze code for vulnerabilities.
Focus on input validation, auth flaws, and data exposure.
Never modify files.
```

The markdown body after the frontmatter **replaces** the default provider system prompt entirely.

### Creating Agents via JSON

Configure agents in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "researcher": {
      "description": "Research agent with custom tools only",
      "mode": "primary",
      "model": "anthropic/claude-sonnet-4-20250514",
      "prompt": "{file:./prompts/researcher.txt}",
      "temperature": 0.2,
      "permission": {
        "edit": "deny",
        "bash": "deny"
      }
    }
  }
}
```

The `{file:./path}` syntax loads prompt text from a file relative to the config location.

### Agent Options Reference

| Option | Type | Description |
|--------|------|-------------|
| `description` | string | When to use this agent (shown to LLM for subagent selection) |
| `mode` | `"primary"` \| `"subagent"` \| `"all"` | How the agent can be used (default: `"all"`) |
| `model` | string | `provider/model-id` override |
| `variant` | string | Model variant (applies only with the agent's configured model) |
| `prompt` | string | System prompt text or `{file:./path}` reference |
| `temperature` | number | 0.0–1.0 randomness control |
| `top_p` | number | 0.0–1.0 response diversity |
| `steps` | number | Max agentic iterations before forcing text-only response |
| `permission` | object | Per-tool permission rules |
| `tools` | object | Legacy tool enable/disable (deprecated; use `permission`) |
| `hidden` | boolean | Hide from `@` autocomplete (subagents only) |
| `disable` | boolean | Disable the agent entirely |
| `color` | string | Hex color or theme color for UI |
| *anything else* | any | Passed through to the provider as model options (e.g. `reasoningEffort`) |

---

## Prompts

### Replacing the System Prompt

When an agent has a `prompt` field, it **replaces** the default provider-specific system prompt. Without it, OpenCode selects a built-in prompt based on model ID (Anthropic, OpenAI, Gemini, etc.).

The assembly logic in `llm.ts`:

```
if agent.prompt exists → use agent.prompt
else → use provider default (anthropic.txt, codex.txt, etc.)
```

### Instruction Files

Additional instructions are appended to every turn automatically:

1. **Project-level**: `AGENTS.md`, `CLAUDE.md`, or `CONTEXT.md` found by walking up from the working directory
2. **Global**: `~/.config/opencode/AGENTS.md` or `~/.claude/CLAUDE.md`
3. **Config `instructions` array**: file paths, globs, or HTTP URLs

```json
{
  "instructions": [
    "./prompts/style-guide.md",
    "~/shared-instructions.md",
    "https://example.com/team-rules.txt"
  ]
}
```

All resolved files are prefixed with `"Instructions from: <path>"` and appended to the system prompt.

### Per-Call System Injection

The SDK `session.prompt` accepts a `system` field that injects additional text for that single turn:

```typescript
await client.session.promptAsync({
  sessionID: id,
  agent: "build",
  system: "Output your response as JSON with keys: analysis, score, issues",
  parts: [{ type: "text", text: "Review the auth module." }],
})
```

### Prompt Assembly Order

The final system prompt sent to the LLM is assembled in this order:

1. **Agent prompt** (or provider default if none)
2. **Environment block** (model ID, working directory, platform, date)
3. **Skills description** (if skill tool is enabled)
4. **Instruction files** (AGENTS.md, config `instructions`, URLs)
5. **Per-call `system` field** (from SDK/API)
6. **Plugin `experimental.chat.system.transform` hook** (can mutate the entire array)

---

## Tools

### Built-in Tools

| Tool ID | Description |
|---------|-------------|
| `bash` | Shell command execution |
| `read` | File reading |
| `glob` | File pattern matching |
| `grep` | Content search |
| `edit` | File editing |
| `write` | File creation/overwrite |
| `task` | Subagent invocation |
| `webfetch` | URL fetching |
| `todowrite` | Todo list management |
| `websearch` | Web search (requires Zen or `OPENCODE_ENABLE_EXA`) |
| `codesearch` | Code search (requires Zen or `OPENCODE_ENABLE_EXA`) |
| `skill` | Load specialized skill instructions |
| `apply_patch` | Patch application (GPT models only) |
| `lsp` | LSP queries (requires `OPENCODE_EXPERIMENTAL_LSP_TOOL`) |
| `batch` | Batch operations (requires `experimental.batch_tool: true`) |

### Disabling Tools via Permissions

The `permission` field provides fine-grained control. Actions are `"allow"`, `"ask"`, or `"deny"`:

```json
{
  "agent": {
    "readonly": {
      "permission": {
        "edit": "deny",
        "bash": {
          "*": "deny",
          "git diff*": "allow",
          "git log*": "allow"
        },
        "webfetch": "deny",
        "task": {
          "*": "deny",
          "explore": "allow"
        }
      }
    }
  }
}
```

The `"deny"` action removes the tool from the LLM entirely — it won't see it or attempt to use it.

For bash, glob patterns match against the command. The **last matching rule wins**.

### Disabling Tools via the tools Key

The legacy `tools` key maps `true` → `{"*": "allow"}` and `false` → `{"*": "deny"}`:

```json
{
  "agent": {
    "minimal": {
      "tools": {
        "read": true,
        "glob": true,
        "grep": true,
        "edit": false,
        "write": false,
        "bash": false,
        "webfetch": false,
        "task": false,
        "skill": false,
        "todowrite": false
      }
    }
  }
}
```

Wildcards work for MCP tool namespaces: `"my_mcp_*": false`.

### Per-Call Tool Overrides

The SDK `session.promptAsync` accepts a `tools` map that overrides for a single turn:

```typescript
await client.session.promptAsync({
  sessionID: id,
  tools: { bash: false, edit: false, "research_search": true },
  parts: [{ type: "text", text: "Search for auth patterns." }],
})
```

### Custom Tools via MCP

MCP servers are configured in `opencode.json` and provide tools without code changes:

```json
{
  "mcp": {
    "research": {
      "type": "local",
      "command": ["node", "./my-mcp-server.js"],
      "environment": { "API_KEY": "{env:RESEARCH_API_KEY}" },
      "enabled": true,
      "timeout": 60000
    },
    "remote-tools": {
      "type": "remote",
      "url": "https://mcp.example.com/sse",
      "headers": { "Authorization": "Bearer {env:MCP_TOKEN}" },
      "enabled": true
    }
  }
}
```

MCP tools are namespaced as `<server>_<tool>` (e.g. `research_search`). You can selectively enable/disable them per agent:

```json
{
  "agent": {
    "researcher": {
      "tools": {
        "research_*": true,
        "remote-tools_*": false
      }
    }
  }
}
```

### Custom Tools via Tool Files

Drop `.ts` or `.js` files in `.opencode/tool/` or `.opencode/tools/`:

```typescript
// .opencode/tools/evaluate.ts
import { tool } from "@opencode-ai/plugin"
import { z } from "zod"

export const evaluate = tool({
  description: "Evaluate code quality and return a score",
  args: {
    code: z.string(),
    criteria: z.array(z.string()),
  },
  async execute(args) {
    // your evaluation logic
    return JSON.stringify({ score: 8, details: "..." })
  },
})
```

### Custom Tools via Plugins

Plugins listed in `config.plugin` can export tools via the `Hooks.tool` map:

```typescript
// my-plugin.ts
import { type Plugin, tool } from "@opencode-ai/plugin"
import { z } from "zod"

const plugin: Plugin = async (input) => ({
  tool: {
    my_tool: tool({
      description: "Custom tool from plugin",
      args: { query: z.string() },
      async execute(args) {
        return "result"
      },
    }),
  },
})

export default plugin
```

---

## Orchestration

### Headless CLI: opencode run

`opencode run` is the non-interactive mode. It sends a single prompt, waits until the session goes idle, and exits.

```bash
# Basic usage
opencode run "Refactor the auth module"

# Pipe input
echo "Fix the bug in auth.ts" | opencode run

# With options
opencode run --agent researcher --model anthropic/claude-sonnet-4-20250514 "Analyze this codebase"

# Continue a previous session
opencode run -c "Now implement the plan"

# Continue a specific session
opencode run -s <session-id> "Next step"

# Fork before continuing
opencode run -c --fork "Try a different approach"

# JSON event stream for scripting
opencode run --format json "Generate a report"

# Run a slash command
opencode run --command my-command "arguments here"
```

In `run` mode, permission prompts are auto-rejected (`reject`) and question/plan tools are denied.

### Server Mode: opencode serve

Start a headless HTTP server for programmatic access:

```bash
opencode serve --port 4096 --hostname 0.0.0.0
```

The server exposes a REST API with SSE for events. Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session` | Create a session |
| `POST` | `/session/:id/message` | Send prompt (blocking) |
| `POST` | `/session/:id/prompt_async` | Send prompt (fire-and-forget, 204) |
| `GET` | `/session/:id/message` | List messages (paginated) |
| `GET` | `/event` | SSE event stream |
| `POST` | `/session/:id/abort` | Abort current generation |
| `POST` | `/session/:id/fork` | Fork a session |
| `GET` | `/doc` | OpenAPI spec |

### SDK Orchestration

The JS SDK (`@opencode-ai/sdk/v2`) wraps the server API. Use `createOpencodeServer` to spawn a headless server and `createOpencodeClient` to interact with it:

```typescript
import { createOpencodeClient, createOpencodeServer } from "@opencode-ai/sdk/v2"

// Provider API keys must be set as environment variables (e.g. OPENAI_API_KEY,
// ANTHROPIC_API_KEY) before spawning the server — the child process inherits them.
const server = await createOpencodeServer({
  config: { model: "openai/gpt-5.4-nano" },
})
const client = createOpencodeClient({ baseUrl: server.url })

const id = (await client.session.create()).data.id

// Send a prompt asynchronously
await client.session.promptAsync({
  sessionID: id,
  agent: "build",
  parts: [{ type: "text", text: "Hello" }],
})

// Poll until the session finishes processing
while (true) {
  await new Promise((r) => setTimeout(r, 500))
  const status = await client.session.status()
  if (!status.data?.[id] || status.data[id].type === "idle") break
}

// Fetch messages to read the assistant's response
const messages = await client.session.messages({ sessionID: id })
// messages.data is an array of { info: Message, parts: Part[] }

server.close()
```

`session.promptAsync` fires the prompt and returns immediately (HTTP 204). The agent loop (tool calls, subagent spawns, etc.) runs in the background. Poll `session.status()` until the session becomes idle, then read messages.

The status map only contains entries for active sessions — when a session's entry is absent or has `type: "idle"`, processing is complete.

**API keys**: `createOpencodeServer` spawns `opencode serve` as a child process, inheriting `process.env`. Provider API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) must be set in the environment before the server starts. Alternatively, use `opencode auth` to store credentials persistently.

### Multi-Step Flows

Sequential prompts on the same session accumulate conversation history:

```typescript
const server = await createOpencodeServer()
const client = createOpencodeClient({ baseUrl: server.url })
const id = (await client.session.create()).data.id

async function send(opts: { agent?: string; system?: string; tools?: Record<string, boolean>; text: string }) {
  await client.session.promptAsync({
    sessionID: id,
    agent: opts.agent,
    system: opts.system,
    tools: opts.tools,
    parts: [{ type: "text", text: opts.text }],
  })
  while (true) {
    await new Promise((r) => setTimeout(r, 500))
    const s = (await client.session.status()).data?.[id]
    if (!s || s.type === "idle") break
  }
}

// Plan
await send({ agent: "plan", text: "Analyze the auth module and create a refactoring plan." })

// Implement
await send({ agent: "build", text: "Implement the plan from the previous messages." })

// Evaluate
await send({
  agent: "plan",
  system: 'Respond with JSON: { "score": <number>, "issues": ["..."] }',
  tools: { bash: false, edit: false },
  text: "Review all changes. Are there any issues?",
})

// Read the last assistant response
const messages = await client.session.messages({ sessionID: id })
const last = messages.data?.findLast((m) => m.info.role === "assistant")
const part = last?.parts.find((p) => p.type === "text" && "text" in p)
const text = part && "text" in part ? part.text : undefined

server.close()
```

Each call can override `agent`, `model`, `tools`, and `system` independently.

### Extracting Trajectories

After a session completes, you can extract the full trajectory — every message, tool call, reasoning step, and token count — for logging, evaluation, or dataset creation:

```typescript
const messages = await client.session.messages({ sessionID: id })

const trajectory = (messages.data ?? []).map((m) => ({
  role: m.info.role,
  agent: (m.info as any).agent,
  model: (m.info as any).modelID,
  text: m.parts
    .filter((p) => p.type === "text" && "text" in p)
    .map((p) => (p as any).text)
    .join(""),
  tools: m.parts
    .filter((p) => p.type === "tool")
    .map((p) => ({
      tool: (p as any).tool,
      status: (p as any).state?.status,
      input: (p as any).state?.input,
      output: (p as any).state?.output?.slice(0, 500),
    })),
  tokens: (m.info as any).tokens,
  cost: (m.info as any).cost,
}))

import { writeFileSync } from "fs"
writeFileSync("trajectory.json", JSON.stringify(trajectory, null, 2))
```

Each message in the trajectory contains:

| Field | Description |
|-------|-------------|
| `info.role` | `"user"` or `"assistant"` |
| `info.agent` | Which agent handled this turn (e.g. `"build"`, `"planner"`) |
| `info.modelID` | Model used (e.g. `"gpt-5-nano"`) |
| `info.tokens` | `{ input, output, reasoning, cache: { read, write } }` |
| `info.cost` | Cost in USD |
| `info.time` | `{ created, completed }` timestamps |
| `parts[].type` | Part type: `"text"`, `"tool"`, `"reasoning"`, `"step-start"`, `"step-finish"`, etc. |

Tool parts (`type: "tool"`) contain the full tool call details:

| Field | Description |
|-------|-------------|
| `tool` | Tool ID (e.g. `"bash"`, `"edit"`, `"read"`) |
| `state.status` | `"pending"`, `"running"`, `"completed"`, or `"error"` |
| `state.input` | Tool arguments as a JSON object |
| `state.output` | Tool result string |
| `state.time` | `{ start, end }` timestamps |

This makes it straightforward to build evaluation datasets, compute per-step costs, or replay agent behavior.

### Subagent Invocation

Within a single prompt, the primary agent can spawn subagents via the `task` tool. You can influence this through:

1. **`@agent` mentions in the prompt text** — nudges the model to invoke that subagent
2. **Agent descriptions** — the LLM reads these to decide which subagent to use
3. **`permission.task` rules** — control which subagents an agent can invoke

```json
{
  "agent": {
    "orchestrator": {
      "permission": {
        "task": {
          "*": "deny",
          "researcher": "allow",
          "implementer": "allow"
        }
      }
    }
  }
}
```

Subagents run as child sessions. The `task` tool returns a `task_id` that can be used to resume the same child session in a later turn.

### Event Stream

For real-time monitoring, use SSE to watch session progress instead of polling:

```typescript
const events = await client.event.subscribe()

// events is an SSE stream with typed events:
// - session.status  (idle, busy, retry)
// - message.updated
// - message.part.updated
// - permission.asked
// - session.idle
```

The `opencode run` CLI uses this pattern internally — it calls `promptAsync` and then monitors the event stream until `session.idle` fires.

---

## Plugins

Plugins provide the deepest customization without modifying source code. Configure them in `opencode.json`:

```json
{
  "plugin": ["./my-plugin.ts", "some-npm-package"]
}
```

Or place them in `.opencode/plugin/` or `.opencode/plugins/`.

### Plugin Hooks Reference

| Hook | Description |
|------|-------------|
| `tool` | Register custom tools (map of ID → ToolDefinition) |
| `event` | React to any bus event |
| `config` | Modify config at load time |
| `chat.message` | Intercept/modify user messages before processing |
| `chat.params` | Modify LLM parameters (temperature, topP, topK, options) |
| `chat.headers` | Add HTTP headers to LLM requests |
| `permission.ask` | Override permission decisions |
| `tool.definition` | Modify tool descriptions and parameter schemas sent to the LLM |
| `tool.execute.before` | Intercept tool args before execution |
| `tool.execute.after` | Modify tool output after execution |
| `shell.env` | Inject environment variables into bash tool |
| `command.execute.before` | Intercept slash command execution |
| `experimental.chat.system.transform` | **Mutate the entire system prompt array** |
| `experimental.chat.messages.transform` | Mutate the message history sent to the LLM |
| `experimental.session.compacting` | Customize or replace the compaction prompt |
| `experimental.text.complete` | Modify completed text output |

The `experimental.chat.system.transform` hook is particularly powerful for research — it receives the fully assembled `system: string[]` and can rewrite it arbitrarily.

---

## Configuration Precedence

Config is merged in this order (lowest to highest priority):

1. Remote `.well-known/opencode` (org defaults)
2. Global config (`~/.config/opencode/opencode.json`)
3. Custom config (`OPENCODE_CONFIG` env var)
4. Project config (`opencode.json` in project root)
5. `.opencode` directory files (agents, commands, plugins, config)
6. Inline config (`OPENCODE_CONFIG_CONTENT` env var)

Per-agent config overrides global config. Per-call SDK overrides override agent config.

---

## Other Config Options

These config options are relevant to agent customization but not covered in the sections above.

### default_agent

Set the default primary agent (falls back to `build`):

```json
{
  "default_agent": "researcher"
}
```

### compaction

Control automatic context compaction behavior:

```json
{
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 4096
  }
}
```

- `auto`: enable automatic compaction when context is full (default: `true`)
- `prune`: enable pruning of old tool outputs (default: `true`)
- `reserved`: token buffer to avoid overflow during compaction

### snapshot

Disable filesystem snapshot tracking (disables undo/revert of file changes):

```json
{
  "snapshot": false
}
```

### formatter

Configure or disable code formatters:

```json
{
  "formatter": {
    "prettier": {
      "command": ["prettier", "--write"],
      "extensions": [".ts", ".tsx", ".js"]
    }
  }
}
```

Set to `false` to disable all formatters.

### lsp

Configure or disable LSP servers:

```json
{
  "lsp": {
    "typescript-language-server": {
      "command": ["typescript-language-server", "--stdio"],
      "extensions": [".ts", ".tsx"]
    }
  }
}
```

Set to `false` to disable all LSP servers.

---

## Complete Example

A research setup with custom agents, MCP tools, replaced prompts, a scripted orchestration loop, and trajectory extraction:

**orchestrator.ts** — a self-contained script that defines config, agents, and prompts inline and runs a plan → implement → evaluate loop:

```typescript
import { createOpencodeClient, createOpencodeServer } from "@opencode-ai/sdk/v2"
import { writeFileSync } from "fs"

const MODEL = "openai/gpt-5.4-nano"

const server = await createOpencodeServer({
  config: {
    model: MODEL,
    mcp: {
      research: {
        type: "local",
        command: ["node", "./tools/research-server.js"],
      },
    },
    instructions: ["./prompts/global-rules.md"],
    agent: {
      planner: {
        description: "Analyzes code and produces structured plans",
        mode: "primary",
        model: MODEL,
        prompt: [
          "You are a senior architect. Analyze the codebase and produce a detailed,",
          "step-by-step implementation plan. Do not make any changes yourself.",
          "Output the plan as a numbered list.",
        ].join("\n"),
        temperature: 0.1,
        permission: {
          edit: "deny",
          bash: { "*": "deny", "git log*": "allow", "git diff*": "allow" },
        },
      },
      implementer: {
        description: "Implements plans by editing code",
        mode: "primary",
        model: MODEL,
        prompt: [
          "You are a senior engineer. Follow the plan from the conversation history",
          "exactly. Make all necessary code changes. Do not skip steps.",
        ].join("\n"),
        permission: {
          edit: "allow",
          bash: "allow",
        },
      },
      evaluator: {
        description: "Reviews changes and scores quality",
        mode: "primary",
        model: MODEL,
        prompt: [
          "You are a code reviewer. Review all changes made in this session.",
          "Check for correctness, edge cases, and style.",
        ].join("\n"),
        temperature: 0.0,
        permission: {
          edit: "deny",
          bash: { "*": "deny", "git diff*": "allow" },
        },
      },
    },
  },
})

const client = createOpencodeClient({ baseUrl: server.url })

async function send(id: string, opts: { agent?: string; system?: string; tools?: Record<string, boolean>; text: string }) {
  await client.session.promptAsync({
    sessionID: id,
    agent: opts.agent,
    system: opts.system,
    tools: opts.tools,
    parts: [{ type: "text", text: opts.text }],
  })
  while (true) {
    await new Promise((r) => setTimeout(r, 500))
    const s = (await client.session.status()).data?.[id]
    if (!s || s.type === "idle") break
  }
}

function lastText(messages: any[]): string {
  const last = messages.findLast((m: any) => m.info.role === "assistant")
  for (const p of last?.parts ?? []) {
    if (p.type === "text" && "text" in p) return p.text
  }
  return "{}"
}

async function run(task: string, retries = 3) {
  const id = (await client.session.create()).data.id

  // Plan
  await send(id, { agent: "planner", text: task })

  for (let attempt = 0; attempt < retries; attempt++) {
    // Implement
    await send(id, { agent: "implementer", text: "Implement the plan. Follow it exactly." })

    // Evaluate
    await send(id, {
      agent: "evaluator",
      system: 'Respond ONLY with JSON: { "score": <1-10>, "issues": ["..."] }',
      text: "Review all changes made in this session.",
    })

    const msgs = (await client.session.messages({ sessionID: id })).data ?? []
    const result = JSON.parse(lastText(msgs))

    if (result.score >= 7) {
      console.log(`Passed with score ${result.score} on attempt ${attempt + 1}`)
      break
    }

    console.log(`Score ${result.score}, retrying. Issues: ${result.issues.join(", ")}`)
    await send(id, { agent: "implementer", text: `Fix these issues:\n${result.issues.join("\n")}` })
  }

  // Extract and save the full trajectory
  const msgs = (await client.session.messages({ sessionID: id })).data ?? []
  const trajectory = msgs.map((m: any) => ({
    role: m.info.role,
    agent: m.info.agent,
    model: m.info.modelID,
    text: m.parts.filter((p: any) => p.type === "text").map((p: any) => p.text).join(""),
    tools: m.parts
      .filter((p: any) => p.type === "tool")
      .map((p: any) => ({
        tool: p.tool,
        status: p.state?.status,
        input: p.state?.input,
        output: p.state?.output?.slice(0, 500),
      })),
    tokens: m.info.tokens,
    cost: m.info.cost,
  }))
  writeFileSync("trajectory.json", JSON.stringify(trajectory, null, 2))
  console.log(`Saved trajectory with ${trajectory.length} messages`)
}

await run("Refactor the authentication module to use JWT tokens")
server.close()
```

The `config` passed to `createOpencodeServer` is serialized as `OPENCODE_CONFIG_CONTENT` under the hood — no temp files or `opencode.json` on disk needed. Run with `OPENAI_API_KEY` set:

```bash
OPENAI_API_KEY=sk-... bun run orchestrator.ts
```

This gives you a plan → implement → evaluate loop with per-step agent switching, custom prompts, controlled tool access, trajectory logging, and programmatic flow control — fully self-contained in a single script.

---

## Workspace-Specific Notes

These notes document configuration patterns used in this workspace's `orchestrator.ts` that are not covered in the general reference above.

### Global Permission Mode

The top-level `config.permission` key sets the default permission mode for all tools across all agents. In headless/automated environments, set it to `"allow"` so tool calls are never blocked by permission prompts:

```typescript
const server = await createOpencodeServer({
  config: {
    model: "openai/gpt-5.4-nano-2026-03-17",
    permission: "allow",
  },
})
```

Without this, the server may pause waiting for interactive permission approval that never comes, causing the session to hang.

### Dynamic Provider Registration

When using a model that is not in OpenCode's default catalog, you must register it explicitly via `config.provider`. The model string must be in `provider/model-id` format (e.g. `openai/gpt-5.4-nano-2026-03-17`), and the provider section must declare the model:

```typescript
const model = "openai/gpt-5.4-nano-2026-03-17"
const [providerID, modelID] = model.split("/")

const server = await createOpencodeServer({
  config: {
    model,
    provider: {
      openai: { models: { [modelID]: {} } },
    },
  },
})
```

If the provider section is omitted for an unrecognized model, OpenCode will return a "Model not found" error.

### Event Stream for Completion Detection

The polling approach shown in the SDK Orchestration section (`session.status()`) can miss state transitions. The event stream approach is more reliable for headless use:

```typescript
const resp = await fetch(`${server.url}/event`, {
  headers: { Accept: "text/event-stream" },
})
const reader = resp.body!.getReader()
const decoder = new TextDecoder()

while (true) {
  const { value } = await reader.read()
  if (!value) continue
  const text = decoder.decode(value)
  for (const line of text.split("\n")) {
    if (!line.startsWith("data:")) continue
    const evt = JSON.parse(line.slice(5).trim())

    // Auto-approve permissions in headless mode
    if (evt.type === "permission.asked") {
      const reqID = evt.properties?.id
      if (reqID) {
        client.permission.reply({ requestID: reqID, reply: "always" })
      }
      continue
    }

    // Check for session completion
    if (evt.properties?.sessionID !== sessionID) continue
    if (evt.type === "session.idle") break   // success
    if (evt.type === "session.error") break  // failure
  }
}
reader.cancel()
```

Key events to handle:

| Event | Meaning |
|-------|---------|
| `session.idle` | Session finished processing successfully |
| `session.error` | Session encountered an error |
| `permission.asked` | A tool needs permission approval (reply with `"always"` in headless mode) |

### Environment Variables

The orchestrator receives its configuration via environment variables set by the Python wrapper:

| Variable | Description |
|----------|-------------|
| `OPENCODE_MODEL` | Provider/model string (e.g. `openai/gpt-5.4-nano-2026-03-17`) |
| `OPENAI_API_KEY` | API key for the configured provider |
| `OPENCODE_TRAJECTORY_PATH` | Path to write the trajectory JSON file |
| `OPENCODE_PORT` | Optional port override for the server |

The working directory is set via the subprocess `cwd` argument, not an environment variable. The spawned `opencode serve` process inherits this CWD.

### Server Startup Reliability

In resource-constrained environments (e.g. Docker containers), the OpenCode server may take longer to start or fail intermittently. The orchestrator uses:

- **60-second timeout** for `createOpencodeServer` (default is much shorter)
- **10 retries** for connecting to the SSE event stream
- **5 full server restart attempts** for `ConnectionRefused`, `Unable to connect`, or `TimeoutError` errors, with exponential backoff
