from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

TOOL_TIMEOUT = 300
STDOUT_MAX_PREFIX_LENGTH = 200
STDOUT_MAX_SUFFIX_LENGTH = 200
STDOUT_MAX_LINE_LENGTH = 2000

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]")
SELF_DESTRUCTIVE_KILL_PATTERNS = [
    re.compile(
        r"\b(?:killall|pkill)\b[^\n;|&]*\b(?:python|python3|bash|sh|zsh|node|ruby|perl)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bkill\b[^\n;|&]*\$\((?:pgrep|pidof)[^)]*\b(?:python|python3|bash|sh|zsh|node|ruby|perl)\b[^)]*\)",
        re.IGNORECASE,
    ),
]


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _is_self_destructive_kill_command(*, command: str) -> bool:
    return any(pattern.search(command) for pattern in SELF_DESTRUCTIVE_KILL_PATTERNS)


def _truncate_output(text: str, label: str) -> tuple[str, bool]:
    lines = text.split("\n")
    truncated_lines = []
    line_truncated = False
    for line in lines:
        if len(line) > STDOUT_MAX_LINE_LENGTH:
            truncated_lines.append(
                line[:STDOUT_MAX_LINE_LENGTH]
                + f"... [truncated, line exceeds {STDOUT_MAX_LINE_LENGTH} chars]"
            )
            line_truncated = True
        else:
            truncated_lines.append(line)

    total = len(truncated_lines)
    max_lines = STDOUT_MAX_PREFIX_LENGTH + STDOUT_MAX_SUFFIX_LENGTH
    if total > max_lines:
        prefix = truncated_lines[:STDOUT_MAX_PREFIX_LENGTH]
        suffix = truncated_lines[-STDOUT_MAX_SUFFIX_LENGTH:]
        omitted = total - max_lines
        return (
            "\n".join(
                prefix
                + [f"\n... [{omitted} lines omitted from {label}] ...\n"]
                + suffix
            ),
            True,
        )
    return "\n".join(truncated_lines), line_truncated


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    command = args.get("command", "")
    cwd = args.get("cwd")
    keep_ansi = args.get("keep_ansi", False)
    env_vars = args.get("env")

    if not command or not command.strip():
        return "Error: Command string is empty or contains only whitespace"
    if _is_self_destructive_kill_command(command=command):
        return (
            "Error: Refusing broad process-kill command because it may terminate the "
            "agent runtime itself. Find the exact service PID first, then kill only "
            "that PID or target a narrowly scoped process name."
        )

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

    stdout, stdout_truncated = _truncate_output(stdout, "stdout")
    stderr, stderr_truncated = _truncate_output(stderr, "stderr")

    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f"STDERR:\n{stderr.rstrip()}")

    if stdout_truncated:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                prefix="forge_shell_stdout_",
                suffix=".txt",
            ) as tmp:
                tmp.write(result.stdout)
                parts.append(f"STDOUT full output: {tmp.name}")
        except Exception:
            pass

    if stderr_truncated:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                prefix="forge_shell_stderr_",
                suffix=".txt",
            ) as tmp:
                tmp.write(result.stderr)
                parts.append(f"STDERR full output: {tmp.name}")
        except Exception:
            pass

    exit_info = f"Exit code: {result.returncode}"
    parts.append(exit_info)

    return "\n".join(parts)
