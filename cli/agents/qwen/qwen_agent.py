from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.interface import AgentRuntimeConfig
from agents.qwen.util import run_subprocess


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


class QwenHeadlessAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        token_limit = _coerce_int(value=options.get("token_limit"), default=131072)
        sampling_params = dict(options.get("sampling_params", {}))
        mcp_servers = dict(options.get("mcp_servers", {}))

        settings: dict[str, Any] = {
            "selectedAuthType": "openai",
            "sessionTokenLimit": token_limit,
        }
        if sampling_params:
            settings["sampling_params"] = sampling_params
        if mcp_servers:
            settings["mcpServers"] = mcp_servers

        env = {
            "OPENAI_MODEL": cfg.model.model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_API_KEY": cfg.model.api_key,
            "PATH": os.environ.get("PATH", ""),
            "NODE_NO_WARNINGS": "1",
            **{key: str(val) for key, val in dict(options.get("env", {})).items()},
        }

        try:
            with tempfile.TemporaryDirectory(prefix="qwen-") as tmp_dir:
                tmp_home = Path(tmp_dir)
                qwen_settings_path = tmp_home / "qwen-settings.json"
                qwen_settings_path.write_text(
                    json.dumps(settings, indent=2),
                    encoding="utf-8",
                )

                # Force stateless execution by isolating all runtime state to temp dirs.
                env["HOME"] = str(tmp_home)
                env["XDG_CONFIG_HOME"] = str(tmp_home / ".config")
                env["XDG_CACHE_HOME"] = str(tmp_home / ".cache")
                env["XDG_STATE_HOME"] = str(tmp_home / ".state")
                env["QWEN_CODE_SYSTEM_SETTINGS_PATH"] = str(qwen_settings_path)
                # Compatibility: older qwen-code used GEMINI_* naming.
                env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(qwen_settings_path)

                run_subprocess(
                    args=["qwen", "-p", instruction, "-y"],
                    cwd=str(Path.cwd()),
                    env=env,
                    check=True,
                )
        except FileNotFoundError:
            console.print(
                Panel(
                    "qwen CLI not found. Install @qwen-code/qwen-code and ensure `qwen` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except (subprocess.CalledProcessError, ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1

        return 0
