# Qwen Agent Fix Plan (Standalone)

## Objective

Make the `qwen` agent in this repo work with:

- OpenRouter Qwen/open-model routes
- OpenAI models when they are compatible with Qwen Code request path
- Local OpenAI-compatible endpoints

and provide explicit failures for unsupported model/endpoint combos.

## Repo Context

- Qwen agent runtime: `cli/agents/qwen/qwen_agent.py`
- Shared compatibility helpers: `cli/agents/openai_compat.py`
- Local wrappers/install:
  - `cli/setup` creates `cli/.bin/qwen`
  - `cli/clean` removes local artifacts
- Tests:
  - `cli/tests/test_agents_audit.py`
  - `cli/tests/test_headless_live.py`

## Current Symptoms (Known)

- Qwen works with OpenRouter Qwen profile.
- Qwen can fail for some GPT models on OpenAI with:
  - `not a chat model ... use v1/completions`
- This is usually endpoint/model capability mismatch, not wrapper process failure.

## External Guidance (Research Targets)

1. Qwen docs on model providers (`openai` auth type / OpenAI-compatible endpoints).
2. OpenRouter model IDs usable through OpenAI-compatible API.
3. Confirm Qwen CLI behavior for non-chat models and fallback options.

## Implementation Tasks

1. **Env contract correctness**
   - Ensure these are always passed:
     - `OPENAI_BASE_URL`
     - `OPENAI_API_KEY`
     - `OPENAI_MODEL`
   - Keep `OPENAI_API_BASE` alias for broader compatibility.
   - Preserve local wrapper binary resolution from `cli/.bin`.

2. **Model normalization**
   - Keep provider-aware normalization in `openai_compat.py`.
   - Avoid destructive transformation that breaks valid OpenRouter IDs.

3. **Compatibility handling**
   - For known non-chat-model errors, convert output to explicit actionable message:
     - suggest compatible chat model
     - suggest `terminus-2` if needed for that model class.
   - Avoid over-restrictive preflight that blocks valid OpenRouter routes.

4. **Qwen settings isolation**
   - Keep qwen stateless settings file behavior (`QWEN_CODE_SYSTEM_SETTINGS_PATH`) unless it causes provider auth issues.

## Debugging Checklist

1. Rebuild wrappers:
   - `./cli/clean && ./cli/setup`
2. Direct qwen probes:
   - `./cli/.bin/qwen --help`
   - run with OpenRouter env + model
   - run with OpenAI env + model
3. Wrapper probes:
   - `./cli/.bin/terminus2-cli --agent qwen --model qwen3.5-35b-a3b "Reply with exactly: OK"`
   - `./cli/.bin/terminus2-cli --agent qwen --model gpt-5.3-codex "Reply with exactly: OK"`
4. Capture output patterns and classify:
   - transport/auth errors
   - model capability mismatch
   - parsing/runtime wrapper bugs

## Verification Requirements

1. Deterministic tests:
   - `uv run --directory cli python -m unittest tests.test_agents_audit`
2. Lint:
   - `uv run --directory cli ruff check agents tests`
3. Smoke acceptance:
   - OpenRouter Qwen profile succeeds.
   - OpenAI compatible model succeeds when model supports chat-completions.
   - Unsupported model class fails with clear message.

## Deliverables

- Updated `cli/agents/qwen/qwen_agent.py`
- Updated helper behavior in `cli/agents/openai_compat.py` (if needed)
- Updated tests in `cli/tests/test_agents_audit.py`
- Final short matrix: model key, backend, qwen result, reason
