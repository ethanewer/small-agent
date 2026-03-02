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
from agents.local_binary import resolve_agent_binary
from agents.openai_compat import (
    normalize_openai_compatible_model,
    preflight_agent_model_compatibility,
)
from agents.qwen.util import run_subprocess


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _qwen_actionable_error_message(
    *,
    model: str,
    process_error: subprocess.CalledProcessError,
) -> str | None:
    stderr = str(process_error.stderr or "")
    stdout = str(process_error.output or "")
    combined = f"{stdout}\n{stderr}".lower()
    is_chat_mismatch = (
        "not a chat model" in combined
        or "use v1/completions" in combined
        or "chat.completions" in combined
        and "not supported" in combined
    )
    if not is_chat_mismatch:
        return None

    return (
        f"Qwen Code cannot use model '{model}' via Chat Completions on this endpoint.\n"
        "Pick a chat-completions-compatible model (for example: gpt-4.1, gpt-4o-mini, "
        "or an OpenRouter qwen/* chat route), or run this model with the "
        "`terminus-2` agent if you need completion-style workflows."
    )


class QwenHeadlessAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        binary = str(
            options.get("binary") or resolve_agent_binary(default_binary="qwen")
        )
        token_limit = _coerce_int(value=options.get("token_limit"), default=131072)
        sampling_params = dict(options.get("sampling_params", {}))
        mcp_servers = dict(options.get("mcp_servers", {}))
        normalized_model = normalize_openai_compatible_model(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        compatibility_error = preflight_agent_model_compatibility(
            agent_key="qwen",
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

        settings: dict[str, Any] = {
            "selectedAuthType": "openai",
            "sessionTokenLimit": token_limit,
        }
        if sampling_params:
            settings["sampling_params"] = sampling_params
        if mcp_servers:
            settings["mcpServers"] = mcp_servers

        env = {
            "OPENAI_MODEL": normalized_model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_API_BASE": cfg.model.api_base,
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
                    args=[binary, "-p", instruction, "-y"],
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
        except subprocess.CalledProcessError as err:
            actionable_error = _qwen_actionable_error_message(
                model=cfg.model.model,
                process_error=err,
            )
            if actionable_error:
                console.print(
                    Panel(
                        actionable_error,
                        title="Agent Compatibility Error",
                        border_style="red",
                    )
                )
                return 1

            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1
        except (ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1

        return 0
