# pyright: reportMissingImports=false, reportMissingTypeArgument=false

from __future__ import annotations

import os
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).parent / "templates"

TOOL_NAME_MAP = {
    "read": "read",
    "write": "write",
    "patch": "patch",
    "fs_search": "fs_search",
    "shell": "shell",
    "fetch": "fetch",
    "remove": "remove",
    "undo": "undo",
    "todo_write": "todo_write",
    "todo_read": "todo_read",
}


def get_environment() -> dict[str, Any]:
    """Detect environment info matching Forge's SystemContext.env."""
    os_name = platform.system().lower()
    if os_name == "darwin":
        os_desc = f"macOS ({platform.machine()})"
    elif os_name == "linux":
        os_desc = f"Linux ({platform.machine()})"
    else:
        os_desc = f"{platform.system()} ({platform.machine()})"

    return {
        "os": os_desc,
        "cwd": os.getcwd(),
        "shell": os.environ.get("SHELL", "/bin/sh"),
        "home": str(Path.home()),
        "max_read_size": 2000,
        "max_line_length": 2000,
        "max_file_size": 262144,
        "max_image_size": 262144,
        "maxReadSize": "2000",
        "maxLineLength": "2000",
        "maxFileSize": "262144",
        "maxImageSize": "262144",
        "stdoutMaxPrefixLength": "200",
        "stdoutMaxSuffixLength": "200",
        "stdoutMaxLineLength": "2000",
        "fetch_truncation_limit": 40000,
        "sem_search_limit": 200,
        "sem_search_top_k": 20,
    }


def list_cwd_files(cwd: str, max_depth: int = 1) -> list[dict[str, Any]]:
    """List files in cwd for the system prompt context."""
    files = []
    cwd_path = Path(cwd)
    try:
        for entry in sorted(cwd_path.iterdir()):
            name = entry.name
            if name.startswith("."):
                continue
            files.append(
                {
                    "path": name,
                    "is_dir": entry.is_dir(),
                }
            )
    except PermissionError:
        pass
    return files


def _load_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    if path.exists():
        return path.read_text()
    return ""


def _render_handlebars(template_str: str, context: dict[str, Any]) -> str:
    """Render a Handlebars template using pybars3."""
    try:
        from pybars import Compiler
    except ImportError:
        return _render_fallback(template_str, context)

    compiler = Compiler()

    partials: dict[str, Any] = {}
    for md_file in TEMPLATES_DIR.glob("*.md"):
        partial_name = md_file.name
        partial_content = md_file.read_text()
        try:
            partials[partial_name] = compiler.compile(partial_content)
        except Exception:
            partials[partial_name] = compiler.compile(
                partial_content.replace("{{else}}", "")
            )

    def not_helper(this, value):
        return not value

    def contains_helper(this, collection, item):
        if isinstance(collection, (list, tuple)):
            return item in collection
        if isinstance(collection, str):
            return item in collection
        return False

    def gt_helper(this, a, b):
        try:
            return float(a) > float(b)
        except (TypeError, ValueError):
            return False

    def eq_helper(this, a, b):
        return str(a) == str(b)

    def inc_helper(this, value):
        try:
            return int(value) + 1
        except (TypeError, ValueError):
            return value

    helpers = {
        "not": not_helper,
        "contains": contains_helper,
        "gt": gt_helper,
        "eq": eq_helper,
        "inc": inc_helper,
    }

    try:
        compiled = compiler.compile(template_str)
        result = compiled(context, helpers=helpers, partials=partials)
        return str(result) if result else ""
    except Exception:
        return _render_fallback(template_str, context)


def _render_fallback(template_str: str, context: dict[str, Any]) -> str:
    """Minimal fallback renderer when pybars3 is not available."""
    result = template_str

    def resolve(path: str, ctx: Any) -> Any:
        parts = path.strip().split(".")
        val = ctx
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, "")
            else:
                return ""
        return val

    result = re.sub(
        r"\{\{>\s*([^}]+)\}\}",
        lambda m: _render_fallback(
            _load_template(m.group(1).strip()),
            context,
        ),
        result,
    )

    result = re.sub(
        r"\{\{#if\s+skills\}\}.*?\{\{else\}\}.*?\{\{/if\}\}",
        "",
        result,
        flags=re.DOTALL,
    )
    result = re.sub(
        r"\{\{#if\s+\(not\s+tool_supported\)\}\}.*?\{\{/if\}\}",
        "",
        result,
        flags=re.DOTALL,
    )

    def replace_if_block(m: re.Match) -> str:
        condition = m.group(1).strip()
        body = m.group(2)
        val = resolve(condition, context)
        if val:
            return _render_fallback(body, context)
        return ""

    result = re.sub(
        r"\{\{#if\s+([^}]+)\}\}(.*?)\{\{/if\}\}",
        replace_if_block,
        result,
        flags=re.DOTALL,
    )

    def replace_each_block(m: re.Match) -> str:
        collection_path = m.group(1).strip()
        body = m.group(2)
        items = resolve(collection_path, context)
        if not isinstance(items, (list, tuple)):
            return ""
        parts = []
        for item in items:
            rendered = body
            rendered = re.sub(
                r"\{\{this\.(\w+)\}\}",
                lambda mm: str(
                    item.get(mm.group(1), "") if isinstance(item, dict) else ""
                ),
                rendered,
            )
            rendered = re.sub(
                r"\{\{(\w+)\}\}",
                lambda mm: str(
                    item.get(mm.group(1), "") if isinstance(item, dict) else ""
                ),
                rendered,
            )
            parts.append(rendered)
        return "".join(parts)

    result = re.sub(
        r"\{\{#each\s+([^}]+)\}\}(.*?)\{\{/each\}\}",
        replace_each_block,
        result,
        flags=re.DOTALL,
    )

    def replace_var(m: re.Match) -> str:
        path = m.group(1).strip()
        val = resolve(path, context)
        return str(val) if val != "" else ""

    result = re.sub(r"\{\{([^#/!>][^}]*?)\}\}", replace_var, result)

    return result


def build_system_prompt(
    env: dict[str, Any],
    files: list[dict[str, Any]],
    tool_names: list[str],
    custom_rules: str = "",
) -> list[str]:
    """Build the two-part system prompt matching Forge's SystemPrompt.

    Returns [agent_template_rendered, custom_agent_template_rendered].
    """
    tool_names_map = {name: name for name in TOOL_NAME_MAP}

    template_context: dict[str, Any] = {
        "env": env,
        "files": files,
        "tool_supported": True,
        "tool_names": tool_names_map,
        "custom_rules": custom_rules,
        "skills": [],
        "model": {"input_modalities": []},
    }

    agent_md = (TEMPLATES_DIR / "forge.md").read_text()
    parts = agent_md.split("---", 2)
    if len(parts) >= 3:
        agent_body = parts[2].strip()
    else:
        agent_body = agent_md

    agent_rendered = _render_handlebars(agent_body, template_context)

    custom_template = _load_template("forge-custom-agent-template.md")
    custom_rendered = _render_handlebars(custom_template, template_context)

    return [agent_rendered, custom_rendered]


def build_user_prompt(event_name: str, event_value: str) -> str:
    """Build the user prompt matching Forge's user_prompt template."""
    current_date = datetime.now().strftime("%A %b %d, %Y")
    return f"<{event_name}>{event_value}</{event_name}>\n<system_date>{current_date}</system_date>"
