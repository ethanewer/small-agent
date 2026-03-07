from __future__ import annotations

import re
from pathlib import Path

_EXEC_RE = re.compile(r'exec\s+"([^"]+)"')


def _wrapper_target_exists(wrapper_path: Path) -> bool:
    try:
        text = wrapper_path.read_text(encoding="utf-8")
    except OSError:
        return False

    match = _EXEC_RE.search(text)
    if not match:
        return False

    target = match.group(1)
    if target.startswith("$") or target.startswith("{"):
        return True

    return Path(target).exists()


def resolve_agent_binary(*, default_binary: str) -> str:
    cli_root = Path(__file__).resolve().parents[1]

    local_wrapper = cli_root / ".local" / "bin" / default_binary
    if local_wrapper.exists() and _wrapper_target_exists(wrapper_path=local_wrapper):
        return str(local_wrapper)

    npm_binary = (
        cli_root / ".local" / "tools" / "node_modules" / ".bin" / default_binary
    )
    if npm_binary.exists():
        return str(npm_binary)

    return default_binary
