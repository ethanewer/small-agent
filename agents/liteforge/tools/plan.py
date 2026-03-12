from __future__ import annotations

from datetime import datetime
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
    plans_dir = cwd / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{current_date}-{plan_name}-{version}.md"
    filepath = plans_dir / filename

    if filepath.exists():
        return (
            "Error: Plan file already exists at "
            f"{filepath}. Use a different plan name or version to avoid conflicts."
        )

    filepath.write_text(content)

    return f"Plan created: {filepath}"
