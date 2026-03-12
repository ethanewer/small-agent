from __future__ import annotations

from pathlib import Path
from typing import Any


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    plan_name = args.get("plan_name", "")
    version = args.get("version", "")
    content = args.get("content", "")

    if not plan_name:
        return "Error: plan_name is required"
    if not version:
        return "Error: version is required"
    if not content:
        return "Error: content is required"

    cwd = Path(env.get("cwd", "."))
    plans_dir = cwd / ".forge" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    safe_name = plan_name.replace(" ", "-").replace("/", "-")
    safe_version = version.replace(" ", "-").replace("/", "-")
    filename = f"{safe_name}-{safe_version}.md"
    filepath = plans_dir / filename

    filepath.write_text(content)

    return f"Plan created: {filepath}"
