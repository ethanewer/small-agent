# Reverse-Engineering Notes (ToolMind-Web-QA)

This note summarizes concrete structure observed from sampled released trajectories and what the harness emulates.

## Sampled observations

Sample window: first 40 rows from `open-wiki-traj.jsonl`.

- Row keys: `key`, `id`, `conversations`
- Turn keys: `role`, `content`, `loss`
- Roles observed: `system`, `user`, `assistant`
- One `system` turn per row at the beginning
- Turn count range in sample: 35 to 609
- Mean turn count in sample: 257.7
- Assistant turns in sample: 5134
- User turns in sample: 5134
- System turns in sample: 40
- `loss` counts in sample:
  - `0`: 9996
  - `1`: 312
  - `loss=1` ratio: ~3.03% of all turns

Assistant content patterns:
- `<think>...</think>` appears in essentially all assistant turns in sampled rows
- `<use_mcp_tool>...</use_mcp_tool>` appears in most assistant turns

Tool usage (sampled assistant tool calls):
- `search_and_scrape_webpage.google_search`: 2734
- `jina_scrape_llm_summary.scrape_and_extract_info`: 1995
- `tool-python.run_python_code`: 242
- `tool-python.create_sandbox`: 37
- `tool-python.download_file_from_internet_to_sandbox`: 45
- `tool-python.run_command`: 1

## Harness fidelity targets

Implemented to match:
- Same row/turn JSON schema
- Same MCP XML call structure
- Same stepwise loop (assistant tool call -> tool JSON result as next user turn)
- Same core tool namespaces and methods
- Sparse critical-turn labeling option for `loss`

Not guaranteed to match:
- The exact proprietary prompts/checkers used by the original pipeline
- The exact model used in generation
- Exact tool backend behavior and reliability
- Exact `loss` assignment logic from turn-level judgment
