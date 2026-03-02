# Claude Agent Fix Plan (Standalone)

## Objective

Make the `claude` agent in this repo work with:

- OpenRouter models (including non-Claude open models such as Qwen/GPT)
- OpenAI models when the Claude Code gateway supports them
- Local OpenAI-compatible endpoints when supported by Claude Code gateway mode

This plan is self-contained and can be executed independently.

## Repo Context

- Wrapper entrypoint: `cli/cli.py`
- Claude agent runtime: `cli/agents/claude/claude_agent.py`
- Shared compatibility helpers: `cli/agents/openai_compat.py`
- Local CLI wrappers/binaries: `cli/.bin` and `cli/.tools` (created by `cli/setup`)
- Setup/clean scripts: `cli/setup`, `cli/clean`
- Tests to update/run:
  - `cli/tests/test_agents_audit.py`
  - `cli/tests/test_headless_live.py`
  - `cli/tests/test_main.py` (regression safety)

## Current Symptoms (Known)

- Claude Code often errors with:
  - `There's an issue with the selected model (...)`
  - or authentication/login style failures depending on env/home isolation
- Prior logic included hard compatibility gating by model family, which should not be used as a universal blocker for OpenRouter gateway use.

## External Guidance (Research Targets)

1. Anthropic Claude Code gateway docs (`ANTHROPIC_BASE_URL`, token behavior)
2. OpenRouter setup references for Claude Code gateway routing
3. Validate env precedence: `ANTHROPIC_AUTH_TOKEN` vs `ANTHROPIC_API_KEY`

## Implementation Tasks

1. **Gateway env correctness**
   - In `claude_agent.py`, ensure environment includes:
     - `ANTHROPIC_BASE_URL` = selected `api_base`
     - `ANTHROPIC_AUTH_TOKEN` = selected API key
     - `ANTHROPIC_MODEL` = selected/normalized model
   - Keep `OPENAI_*` env values if harmless, but Anthropic gateway vars are primary.

2. **Model selection behavior**
   - Verify whether passing `--model` should be:
     - always passed, or
     - optional behind config toggle (eg `pass_model_arg`) if gateway prefers env model.
   - Implement deterministic behavior and document it.

3. **Auth/home isolation**
   - Validate whether temp `HOME` breaks required runtime auth.
   - If it breaks gateway mode, adjust strategy:
     - keep stateless behavior only where safe, or
     - allow opt-out via agent config (eg `isolate_home` flag).

4. **Error clarity**
   - Preserve non-zero return on failures.
   - Improve error text for gateway/model mismatch and auth mismatch.

## Debugging Checklist

Run each command from repo root (`/Users/ethanewer/small-agent`):

1. Rebuild local wrappers:
   - `./cli/clean && ./cli/setup`
2. Direct Claude probes:
   - `./cli/.bin/claude --help`
   - `./cli/.bin/claude -p "Reply with exactly: OK" --output-format text --model <model>`
3. Wrapper probe:
   - `./cli/.bin/terminus2-cli --agent claude --model qwen3.5-35b-a3b "Reply with exactly: OK"`
   - `./cli/.bin/terminus2-cli --agent claude --model gpt-5.3-codex "Reply with exactly: OK"`
4. Capture and compare env actually passed to subprocess in unit tests (patched `run_subprocess`).

## Verification Requirements

1. Deterministic tests:
   - `uv run --directory cli python -m unittest tests.test_agents_audit tests.test_main`
2. Lint:
   - `uv run --directory cli ruff check agents tests`
3. Smoke acceptance:
   - At least one OpenRouter model succeeds through `claude` agent.
   - OpenAI model path either succeeds or fails with explicit actionable reason (not silent/misleading).
4. No regressions in `qwen`/`opencode` tests due to shared helper changes.

## Deliverables

- Updated `cli/agents/claude/claude_agent.py`
- Any needed updates in `cli/agents/openai_compat.py`
- Updated tests in `cli/tests/test_agents_audit.py` and optionally `test_headless_live.py`
- Brief result table: model key, exit code, pass/fail reason
