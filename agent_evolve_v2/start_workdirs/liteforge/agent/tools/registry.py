# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DESCRIPTIONS_DIR = Path(__file__).parent.parent / "templates"

TOOL_DESCRIPTION_FILES = {
    "read": "fs_read.md",
    "write": "fs_write.md",
    "patch": "fs_patch.md",
    "shell": "shell.md",
    "fs_search": "fs_search.md",
    "remove": "fs_remove.md",
    "undo": "fs_undo.md",
    "fetch": "net_fetch.md",
    "todo_write": "todo_write.md",
    "todo_read": "todo_read.md",
}

_DEFAULT_REPO_DESCRIPTIONS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "forge-repo"
    / "crates"
    / "forge_domain"
    / "src"
    / "tools"
    / "descriptions"
)


def _resolve_descriptions_dir() -> Path:
    env_repo = os.environ.get("FORGE_REPO_PATH", "").strip()
    candidates = []
    if env_repo:
        candidates.append(
            Path(env_repo)
            / "crates"
            / "forge_domain"
            / "src"
            / "tools"
            / "descriptions"
        )
    candidates.append(_DEFAULT_REPO_DESCRIPTIONS_DIR)
    # Local machine canonical clone location.
    candidates.append(
        Path("/Users/ethanewer/mycode/forge-repo")
        / "crates"
        / "forge_domain"
        / "src"
        / "tools"
        / "descriptions"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return _DEFAULT_REPO_DESCRIPTIONS_DIR


TOOL_DESCRIPTIONS_DIR = _resolve_descriptions_dir()


def _int_or_null(fmt: str = "int32") -> dict[str, Any]:
    return {"type": "integer", "format": fmt}


def _bool_or_null() -> dict[str, Any]:
    return {"type": "boolean"}


def _string_or_null() -> dict[str, Any]:
    return {"type": "string"}


def _string_array_or_null() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON Schema for each tool's parameters, matching catalog.rs."""
    return {
        "read": {
            "type": "object",
            "title": "FSRead",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "start_line": {
                    **_int_or_null("int32"),
                    "description": (
                        "The line number to start reading from starting from 1 not 0. "
                        "Only provide if the file is too large to read at once"
                    ),
                },
                "end_line": {
                    **_int_or_null("int32"),
                    "description": (
                        "The line number to stop reading at (inclusive). "
                        "Only provide if the file is too large to read at once"
                    ),
                },
                "show_line_numbers": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true, prefixes each line with its line index (starting at 1). "
                        "Defaults to true."
                    ),
                },
            },
            "required": ["file_path"],
        },
        "write": {
            "type": "object",
            "title": "FSWrite",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If set to true, existing files will be overwritten. "
                        "If not set and the file exists, an error will be returned "
                        "with the content of the existing file."
                    ),
                },
            },
            "required": ["file_path", "content"],
        },
        "patch": {
            "type": "object",
            "title": "FSPatch",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)",
                },
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace all occurrences of old_string (default false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        "fs_search": {
            "type": "object",
            "title": "FSSearch",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents.",
                },
                "path": {
                    **_string_or_null(),
                    "description": (
                        "File or directory to search in (rg PATH). "
                        "Defaults to current working directory."
                    ),
                },
                "glob": {
                    **_string_or_null(),
                    "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        'Output mode: "content" shows matching lines (supports -A/-B/-C context, '
                        '-n line numbers, head_limit), "files_with_matches" shows file paths '
                        '(supports head_limit), "count" shows match counts (supports head_limit). '
                        'Defaults to "files_with_matches".'
                    ),
                },
                "-B": {
                    **_int_or_null("uint32"),
                    "minimum": 0,
                    "description": (
                        "Number of lines to show before each match (rg -B). "
                        'Requires output_mode: "content", ignored otherwise.'
                    ),
                },
                "-A": {
                    **_int_or_null("uint32"),
                    "minimum": 0,
                    "description": (
                        "Number of lines to show after each match (rg -A). "
                        'Requires output_mode: "content", ignored otherwise.'
                    ),
                },
                "-C": {
                    **_int_or_null("uint32"),
                    "minimum": 0,
                    "description": (
                        "Number of lines to show before and after each match (rg -C). "
                        'Requires output_mode: "content", ignored otherwise.'
                    ),
                },
                "-n": {
                    **_bool_or_null(),
                    "description": (
                        "Show line numbers in output (rg -n). "
                        'Requires output_mode: "content", ignored otherwise.'
                    ),
                },
                "-i": {
                    **_bool_or_null(),
                    "description": "Case insensitive search (rg -i)",
                },
                "type": {
                    **_string_or_null(),
                    "description": (
                        "File type to search (rg --type). Common types: js, py, rust, go, java, etc. "
                        "More efficient than include for standard file types."
                    ),
                },
                "head_limit": {
                    **_int_or_null("uint32"),
                    "minimum": 0,
                    "description": (
                        'Limit output to first N lines/entries, equivalent to "| head -N". '
                        "Works across all output modes: content (limits output lines), "
                        "files_with_matches (limits file paths), count (limits count entries). "
                        "When unspecified, shows all results from ripgrep."
                    ),
                },
                "offset": {
                    **_int_or_null("uint32"),
                    "minimum": 0,
                    "description": "Skip first N lines/entries before applying head_limit",
                },
                "multiline": {
                    **_bool_or_null(),
                    "description": (
                        "Enable multiline mode where . matches newlines and patterns can span "
                        "lines (rg -U --multiline-dotall). Default: false."
                    ),
                },
            },
            "required": ["pattern"],
        },
        "shell": {
            "type": "object",
            "title": "Shell",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cwd": {
                    **_string_or_null(),
                    "description": (
                        "The working directory where the command should be executed. "
                        "If not specified, defaults to the current working directory from the environment."
                    ),
                },
                "keep_ansi": {
                    "type": "boolean",
                    "description": (
                        "Whether to preserve ANSI escape codes in the output. "
                        "If true, ANSI escape codes will be preserved in the output. "
                        "If false (default), ANSI escape codes will be stripped from the output."
                    ),
                },
                "env": {
                    **_string_array_or_null(),
                    "description": (
                        'Environment variable names to pass to command execution (e.g., ["PATH", "HOME", "USER"]). '
                        "The system automatically reads the specified values and applies them during command execution."
                    ),
                },
                "description": {
                    **_string_or_null(),
                    "description": (
                        "Clear, concise description of what this command does. Recommended to be "
                        "5-10 words for simple commands. For complex commands with pipes or "
                        'multiple operations, provide more context. Examples: "Lists files in '
                        'current directory", "Installs package dependencies", "Compiles Rust '
                        'project with release optimizations".'
                    ),
                },
            },
            "required": ["command"],
        },
        "fetch": {
            "type": "object",
            "title": "NetFetch",
            "description": "Input type for the net fetch tool",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
                "raw": {
                    **_bool_or_null(),
                    "description": "Get raw content without any markdown conversion (default: false)",
                },
            },
            "required": ["url"],
        },
        "remove": {
            "type": "object",
            "title": "FSRemove",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to remove (absolute path required)",
                },
            },
            "required": ["path"],
        },
        "undo": {
            "type": "object",
            "title": "FSUndo",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path of the file to revert to its previous state.",
                },
            },
            "required": ["path"],
        },
        "todo_write": {
            "type": "object",
            "title": "TodoWrite",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Array of todo items to create or update",
                    "items": {
                        "type": "object",
                        "description": "A todo item",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the todo item",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content/description of the todo item",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of the todo",
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
        "todo_read": {
            "type": "object",
            "title": "TodoRead",
            "properties": {},
            "required": [],
        },
    }


def load_tool_description(tool_name: str, context: dict[str, Any] | None = None) -> str:
    """Load and render a tool description from the descriptions directory."""
    filename = TOOL_DESCRIPTION_FILES.get(tool_name)
    if not filename:
        return ""

    desc_path = TOOL_DESCRIPTIONS_DIR / filename
    if not desc_path.exists():
        return ""

    text = desc_path.read_text()

    if context:
        text = _render_simple_handlebars(text, context)

    return text.strip()


def _render_simple_handlebars(template: str, ctx: dict[str, Any]) -> str:
    """Minimal Handlebars-like variable substitution for tool descriptions."""
    import re

    def replace_var(m: re.Match[str]) -> str:
        path = m.group(1).strip()
        parts = path.split(".")
        val: Any = ctx
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, "")
            else:
                return m.group(0)
        return str(val)

    result = re.sub(r"\{\{([^#/!>][^}]*?)\}\}", replace_var, template)

    result = re.sub(
        r"\{\{#if\s+\(contains\s+[^)]+\)\}\}.*?\{\{/if\}\}",
        "",
        result,
        flags=re.DOTALL,
    )

    return result


def build_tool_definitions(
    tool_names: list[str],
    description_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions for the given tool names."""
    schemas = get_tool_schemas()
    definitions = []

    for name in tool_names:
        schema = schemas.get(name)
        if not schema:
            continue

        description = load_tool_description(name, description_context)

        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )

    return definitions


ALL_TOOL_NAMES = [
    "read",
    "write",
    "patch",
    "fs_search",
    "shell",
    "fetch",
    "remove",
    "undo",
    "todo_write",
    "todo_read",
]

READONLY_TOOL_NAMES = [
    "read",
    "fs_search",
    "fetch",
    "todo_write",
    "todo_read",
]
