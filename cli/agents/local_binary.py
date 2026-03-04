from __future__ import annotations

from pathlib import Path


def resolve_agent_binary(*, default_binary: str) -> str:
    cli_root = Path(__file__).resolve().parents[1]
    local_binary = cli_root / ".local" / "bin" / default_binary
    if local_binary.exists():
        return str(local_binary)

    return default_binary
