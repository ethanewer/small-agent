from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.claude.util import run_subprocess
from agents.interface import AgentRuntimeConfig
from agents.local_binary import resolve_agent_binary
from agents.openai_compat import (
    normalize_openai_compatible_model,
    preflight_agent_model_compatibility,
)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    return bool(value)


def _claude_failure_message(
    *,
    err: subprocess.CalledProcessError,
    api_base: str,
    normalized_model: str,
    pass_model_arg: bool,
    isolate_home: bool,
) -> str:
    return (
        f"claude CLI exited with status {err.returncode}.\n\n"
        f"Gateway env used:\n"
        f"- ANTHROPIC_BASE_URL={api_base}\n"
        f"- ANTHROPIC_MODEL={normalized_model}\n"
        f"- ANTHROPIC_AUTH_TOKEN=[set]\n\n"
        "Troubleshooting:\n"
        f"- Model routing issue: try setting `agents.claude.pass_model_arg` to "
        f"`{str(not pass_model_arg).lower()}`.\n"
        "- Auth issue: verify the selected API key has access for the target gateway/model.\n"
        f"- Auth/session mismatch: set `agents.claude.isolate_home` to "
        f"`{str(not isolate_home).lower()}` if your environment needs shared CLI state.\n\n"
        f"Original error: {err}"
    )


class ClaudeCodeAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        binary = str(
            options.get("binary") or resolve_agent_binary(default_binary="claude")
        )
        output_format = str(options.get("output_format", "text"))
        skip_permissions = _coerce_bool(options.get("skip_permissions"), True)
        pass_model_arg = _coerce_bool(options.get("pass_model_arg"), True)
        isolate_home = _coerce_bool(options.get("isolate_home"), False)
        allowed_tools = list(options.get("allowed_tools", []))
        normalized_model = normalize_openai_compatible_model(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        compatibility_error = preflight_agent_model_compatibility(
            agent_key="claude",
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        if compatibility_error:
            console.print(
                Panel(
                    compatibility_error,
                    title="Agent Compatibility Error",
                    border_style="red",
                )
            )
            return 1

        args = [binary, "-p", instruction, "--output-format", output_format]
        if pass_model_arg:
            args.extend(["--model", normalized_model])
        if skip_permissions:
            args.append("--dangerously-skip-permissions")
        for tool in allowed_tools:
            args.extend(["--allowedTools", str(tool)])

        env = dict(os.environ)
        env.update(
            {
                "OPENAI_MODEL": normalized_model,
                "OPENAI_BASE_URL": cfg.model.api_base,
                "OPENAI_API_BASE": cfg.model.api_base,
                "OPENAI_API_KEY": cfg.model.api_key,
                "ANTHROPIC_MODEL": normalized_model,
                "ANTHROPIC_AUTH_TOKEN": cfg.model.api_key,
                "ANTHROPIC_API_KEY": cfg.model.api_key,
                "ANTHROPIC_BASE_URL": cfg.model.api_base,
                "PATH": os.environ.get("PATH", ""),
                "NODE_NO_WARNINGS": "1",
                **{key: str(val) for key, val in dict(options.get("env", {})).items()},
            }
        )

        try:
            if isolate_home:
                with tempfile.TemporaryDirectory(prefix="claude-headless-") as tmp_dir:
                    tmp_home = Path(tmp_dir)
                    env["HOME"] = str(tmp_home)
                    env["XDG_CONFIG_HOME"] = str(tmp_home / ".config")
                    env["XDG_CACHE_HOME"] = str(tmp_home / ".cache")
                    env["XDG_STATE_HOME"] = str(tmp_home / ".state")

                    run_subprocess(
                        args=args,
                        cwd=str(Path.cwd()),
                        env=env,
                        check=True,
                    )
            else:
                run_subprocess(
                    args=args,
                    cwd=str(Path.cwd()),
                    env=env,
                    check=True,
                )
        except FileNotFoundError:
            console.print(
                Panel(
                    "claude CLI not found. Install @anthropic-ai/claude-code and ensure `claude` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except subprocess.CalledProcessError as err:
            console.print(
                Panel(
                    _claude_failure_message(
                        err=err,
                        api_base=cfg.model.api_base,
                        normalized_model=normalized_model,
                        pass_model_arg=pass_model_arg,
                        isolate_home=isolate_home,
                    ),
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except (ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1

        return 0
