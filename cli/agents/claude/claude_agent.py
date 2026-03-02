from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agents.claude.util import run_subprocess
from agents.interface import AgentRuntimeConfig
from agents.local_binary import resolve_agent_binary
from agents.openai_compat import (
    normalize_openai_compatible_model,
    preflight_agent_model_compatibility,
)


class ClaudeCodeAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        binary = str(
            options.get("binary") or resolve_agent_binary(default_binary="claude")
        )
        output_format = str(options.get("output_format", "text"))
        skip_permissions = bool(options.get("skip_permissions", True))
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
        args.extend(["--model", normalized_model])
        if skip_permissions:
            args.append("--dangerously-skip-permissions")
        for tool in allowed_tools:
            args.extend(["--allowedTools", str(tool)])

        env = {
            "OPENAI_MODEL": normalized_model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_API_BASE": cfg.model.api_base,
            "OPENAI_API_KEY": cfg.model.api_key,
            "ANTHROPIC_API_KEY": cfg.model.api_key,
            "ANTHROPIC_BASE_URL": cfg.model.api_base,
            "PATH": os.environ.get("PATH", ""),
            "NODE_NO_WARNINGS": "1",
            **{key: str(val) for key, val in dict(options.get("env", {})).items()},
        }

        try:
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
        except FileNotFoundError:
            console.print(
                Panel(
                    "claude CLI not found. Install @anthropic-ai/claude-code and ensure `claude` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except (subprocess.CalledProcessError, ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1

        return 0
