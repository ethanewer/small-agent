# ToolMind-Style Trajectory Harness

This is a local harness to generate trajectories that are structurally close to `Nanbeige/ToolMind-Web-QA` `open-wiki-traj.jsonl`.

It reproduces:
- row schema: `{"key","id","conversations"}`
- turn schema: `{"role","content"}`
- MCP XML tool-call format (`<use_mcp_tool> ... </use_mcp_tool>`)
- one tool call per assistant step
- tool-result-as-next-user-message loop

It uses the same 3 MCP server families seen in trajectories:
- `tool-python`
- `search_and_scrape_webpage` (`google_search`)
- `jina_scrape_llm_summary` (`scrape_and_extract_info`)

## What is approximate vs exact

Close to dataset:
- protocol and prompt structure
- turn-level conversation loop
- tool names, argument shape, JSON-style tool responses

Not guaranteed exact:
- original model weights/prompts behind the released data
- exact internal tool backends/routing

## Requirements

- Python 3.9+
- `OPENAI_API_KEY` for the reasoning model
- `SERPER_API_KEY` for high-fidelity `google_search` (recommended)
- optional `JINA_BASE_URL` / `JINA_API_KEY` for Jina-backed scraping
- optional `EXTRACTOR_MODEL` (defaults to `--model`) for extraction summaries

By default, if `SERPER_API_KEY` is missing, `google_search` returns an error (closer to production behavior). You can opt into fallback search with `--allow-fallback-search`.

## Quick start

```bash
cd cli/agents/toolmind_harness
export OPENAI_API_KEY="..."
python3 harness.py \
  --question "What rock band released the 2001 album Lateralus? Wrap final answer in \\boxed{}." \
  --output "./generated_traj_example.json" \
  --key "wikiQA-en-local" \
  --id "newid_local_1" \
  --max-assistant-turns 40 \
  --strict-protocol \
  --min-tool-turns 8
```

## Output format

Example:

```json
{
  "key": "wikiQA-en-local",
  "id": "newid_local_1",
  "conversations": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"<think>...</think> ... <use_mcp_tool>...</use_mcp_tool>"},
    {"role":"user","content":"{ ... tool result json ... }"}
  ]
}
```

## Protocol fidelity controls

- `--strict-protocol`: enforces one MCP tool call per assistant turn
- `--min-tool-turns N`: requires at least `N` tool turns before accepting a final non-tool answer
- `--repair-attempts N`: max consecutive repair loops when format rules are violated
- `--allow-fallback-search`: enable DuckDuckGo fallback if Serper is missing
- `--force-think-tag`: inject `<think>` when the model omits it
- `--request-reasoning` / `--no-request-reasoning`: request separate reasoning fields from the API and inject into `<think>` (default on)
- `--internal-protocol-retry` / `--no-internal-protocol-retry`: retry malformed protocol responses internally (default on)
- `--max-internal-protocol-retries N`: hidden retry count per turn for malformed protocol responses
- `--record-protocol-repairs` / `--no-record-protocol-repairs`: whether to include protocol-repair user turns in saved trajectories (default off)

## Notes on fidelity

Observed in sampled released rows:
- Every row starts with one long system prompt.
- Assistant messages almost always include `<think>...</think>`.
- Most assistant turns are tool-calling turns.
- Tool failures can occur and are returned as structured JSON.
- Tool mix is dominated by:
  - `search_and_scrape_webpage.google_search`
  - `jina_scrape_llm_summary.scrape_and_extract_info`
  - occasional `tool-python.*`

This harness encodes those constraints directly.

## Reasoning-output compatibility

Some reasoning models (including OpenRouter-hosted models) may return reasoning in a separate field (e.g., `message.reasoning`) instead of inside `message.content`.  
The harness now requests reasoning output by default and folds it into the trajectory as:

`<think> ...reasoning... </think>` + assistant visible content.

## Alignment evaluator

Use `alignment_eval.py` for static alignment checks:

```bash
cd cli/agents/toolmind_harness
python3 alignment_eval.py --report "./alignment_report.md"
```

Optional model-in-the-loop checks:

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... \
python3 alignment_eval.py --run-generation --max-assistant-turns 24
```
