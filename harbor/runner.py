"""Minimal runner script for Docker sandboxes.

Reads Config fields from environment variables, calls agents.run(),
and exits with the result code. Used by both Harbor benchmarks and
agent_evolve benchmarks.
"""

from __future__ import annotations

import base64
import json
import os
import sys


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None

    return float(value)


def _maybe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None

    return int(value)


def _maybe_json(value: str | None) -> dict[str, object] | None:
    if value is None or value == "":
        return None

    decoded = base64.b64decode(value).decode()
    result = json.loads(decoded)
    if isinstance(result, dict):
        return result

    return None


def main() -> int:
    from agents import Config, run

    config = Config(
        model=os.environ["CFG_MODEL"],
        api_base=os.environ["CFG_API_BASE"],
        api_key=os.environ["CFG_API_KEY"],
        temperature=_maybe_float(os.environ.get("CFG_TEMPERATURE")),
        context_length=_maybe_int(os.environ.get("CFG_CONTEXT_LENGTH")),
        extra_params=_maybe_json(os.environ.get("CFG_EXTRA_PARAMS_B64")),
        max_turns=int(os.environ.get("CFG_MAX_TURNS", "50")),
        max_wait_seconds=float(os.environ.get("CFG_MAX_WAIT_SECONDS", "60")),
        final_message_enabled=os.environ.get("CFG_FINAL_MESSAGE", "1") == "1",
    )
    instruction = sys.argv[1]
    result = run(instruction=instruction, config=config)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
