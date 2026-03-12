from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

TOOL_TIMEOUT = 300
STDOUT_MAX_PREFIX_LENGTH = 200
STDOUT_MAX_SUFFIX_LENGTH = 200
STDOUT_MAX_LINE_LENGTH = 2000

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _truncate_output(text: str, label: str) -> str:
    lines = text.split("\n")
    truncated_lines = []
    for line in lines:
        if len(line) > STDOUT_MAX_LINE_LENGTH:
            truncated_lines.append(
                line[:STDOUT_MAX_LINE_LENGTH]
                + f"... [truncated, line exceeds {STDOUT_MAX_LINE_LENGTH} chars]"
            )
        else:
            truncated_lines.append(line)

    total = len(truncated_lines)
    max_lines = STDOUT_MAX_PREFIX_LENGTH + STDOUT_MAX_SUFFIX_LENGTH
    if total > max_lines:
        prefix = truncated_lines[:STDOUT_MAX_PREFIX_LENGTH]
        suffix = truncated_lines[-STDOUT_MAX_SUFFIX_LENGTH:]
        omitted = total - max_lines
        return "\n".join(
            prefix + [f"\n... [{omitted} lines omitted from {label}] ...\n"] + suffix
        )
    return "\n".join(truncated_lines)


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    command = args.get("command", "")
    cwd = args.get("cwd")
    keep_ansi = args.get("keep_ansi", False)
    env_vars = args.get("env")

    if not command or not command.strip():
        return "Error: command is required and cannot be empty"

    work_dir = cwd or str(env.get("cwd", "."))
    work_path = Path(work_dir)
    if not work_path.is_absolute():
        work_path = Path(env.get("cwd", ".")) / work_path

    if not work_path.exists():
        return f"Error: Working directory does not exist: {work_path}"

    shell_path = env.get("shell", "/bin/sh")

    proc_env = os.environ.copy()
    if env_vars:
        for var_name in env_vars:
            val = os.environ.get(var_name)
            if val is not None:
                proc_env[var_name] = val

    try:
        result = subprocess.run(
            [shell_path, "-c", command],
            capture_output=True,
            text=True,
            cwd=str(work_path),
            timeout=TOOL_TIMEOUT,
            env=proc_env,
        )
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {TOOL_TIMEOUT} seconds"
    except FileNotFoundError:
        return f"Error: Shell not found: {shell_path}"
    except Exception as e:
        return f"Error: Failed to execute command: {e}"

    stdout = result.stdout
    stderr = result.stderr

    if not keep_ansi:
        stdout = _strip_ansi(stdout)
        stderr = _strip_ansi(stderr)

    stdout = _truncate_output(stdout, "stdout")
    stderr = _truncate_output(stderr, "stderr")

    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f"STDERR:\n{stderr.rstrip()}")

    exit_info = f"Exit code: {result.returncode}"
    parts.append(exit_info)

    return "\n".join(parts)
