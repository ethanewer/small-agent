# OpenCode Agent Fix Plan (Standalone)

## Objective

Make the `opencode` agent in this repo work reliably with:

- OpenRouter models
- OpenAI models (when model IDs are valid in OpenCode provider catalog)
- Local OpenAI-compatible endpoints

and ensure failures are surfaced as non-zero (no false success).

## Repo Context

- OpenCode agent runtime: `cli/agents/opencode/opencode_agent.py`
- OpenCode subprocess utility: `cli/agents/opencode/util.py`
- Shared compat helpers: `cli/agents/openai_compat.py`
- Local wrappers/install:
  - `cli/setup` creates `cli/.bin/opencode`
  - `cli/clean` removes local artifacts
- Tests:
  - `cli/tests/test_agents_audit.py`
  - `cli/tests/test_headless_live.py`

## Current Symptoms (Known)

- OpenCode may return exit `0` while output contains model errors.
- `--model provider/model` combinations can fail with `ProviderModelNotFoundError`.
- OpenCode can ignore wrapper-selected model if provider config resolution is inconsistent.

## External Guidance (Research Targets)

1. OpenCode docs:
   - CLI `--model` expects `provider/model`
   - `opencode models` for discoverable IDs
   - config via `OPENCODE_CONFIG_CONTENT` / `OPENCODE_CONFIG`
2. OpenRouter provider handling in OpenCode config.

## Implementation Tasks

1. **Model source strategy**
   - Keep env-based default model routing using:
     - `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
   - Inject `OPENCODE_CONFIG_CONTENT` to set model/provider explicitly at runtime.
   - Keep optional `pass_model_arg` override for explicit CLI `--model`.

2. **Provider/model normalization**
   - Use shared helper to compute provider-aware model IDs.
   - Ensure OpenRouter uses correct provider namespace expected by OpenCode.
   - Ensure OpenAI/local routes use valid provider namespace and raw model IDs when needed.

3. **False-success prevention**
   - In `opencode/util.py`, keep/extend fail-pattern detection for:
     - `ProviderModelNotFoundError`
     - `Model not found:`
     - `is not a valid model ID`
   - Raise `CalledProcessError` when these patterns appear even if process exits `0`.

4. **Debug visibility**
   - Keep error output actionable in panel text.
   - Include hint to run `opencode models --refresh` when catalog mismatch is detected.

## Debugging Checklist

1. Rebuild local binaries:
   - `./cli/clean && ./cli/setup`
2. Inspect available model IDs:
   - `./cli/.bin/opencode models --refresh`
   - `./cli/.bin/opencode models openrouter`
   - `./cli/.bin/opencode models openai`
3. Direct run probes:
   - `./cli/.bin/opencode run "Reply with exactly: OK" --format default`
   - Optional explicit model:
     - `./cli/.bin/opencode run "Reply with exactly: OK" --model <provider/model> --format default`
4. Wrapper probe:
   - `./cli/.bin/terminus2-cli --agent opencode --model qwen3.5-35b-a3b "Reply with exactly: OK"`
   - `./cli/.bin/terminus2-cli --agent opencode --model gpt-5.3-codex "Reply with exactly: OK"`

## Verification Requirements

1. Deterministic tests:
   - `uv run --directory cli python -m unittest tests.test_agents_audit`
2. Lint:
   - `uv run --directory cli ruff check agents tests`
3. Smoke acceptance:
   - OpenRouter path succeeds for a model ID that exists in OpenCode catalog.
   - OpenAI path succeeds for a valid catalog model.
   - Invalid model IDs fail non-zero with clear reason (no false pass).

## Deliverables

- Updated `cli/agents/opencode/opencode_agent.py`
- Updated `cli/agents/opencode/util.py` (if needed)
- Updated helper logic in `cli/agents/openai_compat.py` (if needed)
- Updated tests in `cli/tests/test_agents_audit.py`
- Short model compatibility table with exact provider/model IDs used
